#!/bin/sh
# nginx/entrypoint.sh — render the site config(s) for this server's domain(s),
# make sure each has a certificate to serve, then run nginx in the foreground.
#
#   DOMAIN         public hostname (from the server's .env via docker-compose);
#                  default datahub.afyaanalytics.com (production)
#   WANDB_DOMAIN   optional: subdomain that serves W&B Server (wandb-local),
#                  e.g. wandb.datahub.afyaanalytics.com — see
#                  templates/wandb.template. Empty = W&B not published.
#   CERTBOT_EMAIL  Let's Encrypt account email, used only when this server has
#                  no certificate yet for a domain
#
# First boot for a domain (no /etc/letsencrypt/live/<domain> yet): nginx
# starts on a throw-away self-signed certificate so it can answer the HTTP-01
# challenge on port 80, certbot obtains the real certificate through the
# webroot, and nginx is switched over and reloaded — no manual step. Each
# domain is handled on its own: a failing W&B certificate (e.g. DNS not set
# yet) never affects the main site.
#
# Renewal: only certificates whose renewal config uses the webroot
# authenticator (i.e. the ones issued by this script) are renewed here, twice
# a day. A certificate issued some other way (production's main one, today)
# is left exactly as it was being handled before.
set -eu

DOMAIN="${DOMAIN:-datahub.afyaanalytics.com}"
WANDB_DOMAIN="${WANDB_DOMAIN:-}"
WEBROOT=/var/www/certbot

log() { echo "nginx-entrypoint: $*"; }

render() {  # $1 = domain, $2 = template name, $3 = site name, $4 = dir with fullchain.pem + privkey.pem
  DOMAIN="$1" SSL_CERT="$4/fullchain.pem" SSL_KEY="$4/privkey.pem" \
    envsubst '${DOMAIN} ${SSL_CERT} ${SSL_KEY}' < "/etc/nginx/templates/$2" > "/etc/nginx/sites-enabled/$3"
  cp "/etc/nginx/sites-enabled/$3" "/etc/nginx/sites-available/$3"
}

wait_for_nginx() {
  for _ in $(seq 1 30); do
    [ -s /run/nginx.pid ] && kill -0 "$(cat /run/nginx.pid)" 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

obtain_certificate() {  # $1 = domain, $2 = template, $3 = site
  wait_for_nginx || { log "nginx did not start; not requesting a certificate for $1"; return 1; }
  log "requesting a Let's Encrypt certificate for $1"
  if certbot certonly --webroot -w "$WEBROOT" -d "$1" \
       --email "$CERTBOT_EMAIL" --agree-tos --non-interactive --keep-until-expiring; then
    render "$1" "$2" "$3" "/etc/letsencrypt/live/$1"
    nginx -t && nginx -s reload && log "now serving the Let's Encrypt certificate for $1"
  else
    log "certificate request for $1 FAILED (is DNS for $1 pointing here and port 80 open?) — still on the self-signed one"
    return 1
  fi
}

setup_site() {  # $1 = domain, $2 = template, $3 = site
  live="/etc/letsencrypt/live/$1"
  boot="/etc/nginx/bootstrap-ssl/$1"
  if [ -s "$live/fullchain.pem" ] && [ -s "$live/privkey.pem" ]; then
    log "using Let's Encrypt certificate for $1"
    render "$1" "$2" "$3" "$live"
    return 0
  fi
  log "no certificate for $1 yet — starting on a temporary self-signed one"
  if [ ! -s "$boot/fullchain.pem" ]; then
    mkdir -p "$boot"
    openssl req -x509 -nodes -newkey rsa:2048 -days 30 -subj "/CN=$1" \
      -keyout "$boot/privkey.pem" -out "$boot/fullchain.pem" 2>/dev/null
  fi
  render "$1" "$2" "$3" "$boot"
  if [ -n "${CERTBOT_EMAIL:-}" ]; then
    obtain_certificate "$1" "$2" "$3" &
  else
    log "CERTBOT_EMAIL is not set — not requesting a certificate for $1"
  fi
}

renew_loop() {  # $@ = domains
  while :; do
    sleep 43200
    for d in "$@"; do
      if grep -qs '^authenticator = webroot' "/etc/letsencrypt/renewal/$d.conf"; then
        certbot renew --cert-name "$d" --quiet --deploy-hook "nginx -s reload" \
          || log "renewal attempt for $d failed; will retry in 12 h"
      fi
    done
  done
}

mkdir -p "$WEBROOT" /etc/nginx/sites-enabled /etc/nginx/sites-available
rm -f /etc/nginx/sites-enabled/wandb   # re-created below only while WANDB_DOMAIN is set

setup_site "$DOMAIN" django_project.template django_project
if [ -n "$WANDB_DOMAIN" ] && [ "$WANDB_DOMAIN" != "$DOMAIN" ]; then
  setup_site "$WANDB_DOMAIN" wandb.template wandb
  renew_loop "$DOMAIN" "$WANDB_DOMAIN" &
else
  renew_loop "$DOMAIN" &
fi

nginx -t
exec nginx -g 'daemon off;'
