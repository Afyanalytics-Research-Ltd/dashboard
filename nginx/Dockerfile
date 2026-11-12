FROM ubuntu:latest

# nginx + certbot (webroot issuing/renewal, see entrypoint.sh); gettext-base
# for envsubst, openssl for the first-boot self-signed certificate.
RUN apt-get update && \
    apt-get install -y nginx python3-certbot-nginx gettext-base openssl && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /etc/nginx/sites-enabled/default

# The site config is a template rendered at start for this server's DOMAIN.
COPY templates/ /etc/nginx/templates
COPY snippets/ /etc/nginx/snippets
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
