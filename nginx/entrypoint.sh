#!/bin/sh
# nginx/entrypoint.sh — render the site config for this server's domain, make
# sure there is a certificate to serve, then run nginx in the foreground.
#
#   DOMAIN         public hostname (from the server's .env via docker-compose);
#                  default datahub.afyaanalytics.com (production)
#   CERTBOT_EMAIL  Let's Encrypt account email, used only when this server has
#                  no certificate yet for DOMAIN
#
# First boot on a new server (no /etc/letsencrypt/live/$DOMAIN yet): nginx
# starts on a throw-away self-signed certificate so it can answer the HTTP-01
# challenge on port 80, certbot obtains the real certificate through the
# webroot, and nginx is switched over and reloaded — no manual step.
#
# Renewal: only certificates whose renewal config uses the webroot
# authenticator (i.e. the ones issued by this script) are renewed here, twice
# a day. A certificate issued some other way (production's, today) is left
# exactly as it was being handled before.
set -eu

DOMAIN="${DOMAIN:-datahub.afyaanalytics.com}"
LIVE="/etc/letsencrypt/live/$DOMAIN"
BOOT="/etc/nginx/bootstrap-ssl/$DOMAIN"
WEBROOT=/var/www/certbot
TEMPLATE=/etc/nginx/templates/django_project.template
SITE=/etc/nginx/sites-enabled/django_project

log() { echo "nginx-entrypoint: $*"; }

render() {  # $1 = directory holding fullchain.pem + privkey.pem
  DOMAIN="$DOMAIN" SSL_CERT="$1/fullchain.pem" SSL_KEY="$1/privkey.pem" \
    envsubst '${DOMAIN} ${SSL_CERT} ${SSL_KEY}' < "$TEMPLATE" > "$SITE"
  cp "$SITE" /etc/nginx/sites-available/django_project
}

wait_for_nginx() {
  for _ in $(seq 1 30); do
    [ -s /run/nginx.pid ] && kill -0 "$(cat /run/nginx.pid)" 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

obtain_certificate() {
  wait_for_nginx || { log "nginx did not start; not requesting a certificate"; return 1; }
  log "requesting a Let's Encrypt certificate for $DOMAIN"
  if certbot certonly --webroot -w "$WEBROOT" -d "$DOMAIN" \
       --email "$CERTBOT_EMAIL" --agree-tos --non-interactive --keep-until-expiring; then
    render "$LIVE"
    nginx -t && nginx -s reload && log "now serving the Let's Encrypt certificate for $DOMAIN"
  else
    log "certificate request FAILED (is DNS for $DOMAIN pointing here and port 80 open?) — still on the self-signed one"
    return 1
  fi
}

renew_loop() {
  while :; do
    sleep 43200
    if grep -qs '^authenticator = webroot' "/etc/letsencrypt/renewal/$DOMAIN.conf"; then
      certbot renew --cert-name "$DOMAIN" --quiet --deploy-hook "nginx -s reload" \
        || log "renewal attempt failed; will retry in 12 h"
    fi
  done
}

mkdir -p "$WEBROOT" /etc/nginx/sites-enabled /etc/nginx/sites-available

if [ -s "$LIVE/fullchain.pem" ] && [ -s "$LIVE/privkey.pem" ]; then
  log "using Let's Encrypt certificate for $DOMAIN"
  render "$LIVE"
else
  log "no certificate for $DOMAIN yet — starting on a temporary self-signed one"
  if [ ! -s "$BOOT/fullchain.pem" ]; then
    mkdir -p "$BOOT"
    openssl req -x509 -nodes -newkey rsa:2048 -days 30 -subj "/CN=$DOMAIN" \
      -keyout "$BOOT/privkey.pem" -out "$BOOT/fullchain.pem" 2>/dev/null
  fi
  render "$BOOT"
  if [ -n "${CERTBOT_EMAIL:-}" ]; then
    obtain_certificate &
  else
    log "CERTBOT_EMAIL is not set — not requesting a certificate"
  fi
fi

renew_loop &

nginx -t
exec nginx -g 'daemon off;'
