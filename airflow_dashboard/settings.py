"""
Afya DataHub - Django Settings
Healthcare Analytics Platform
"""

import os
import logging.handlers
from pathlib import Path
from dotenv import load_dotenv
from django.contrib.messages import constants as messages

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-change-me-in-production')
DEBUG = os.getenv('DEBUG', 'False').strip().lower() in ('true', '1', 'yes', 'on')

# settings.py — add/update these

ALLOWED_HOSTS = [
    'datahub.afyaanalytics.com',
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    '2c97-129-222-187-199.ngrok-free.app',
    '8fa2-105-164-5-177.ngrok-free.app'
]

CSRF_TRUSTED_ORIGINS = [
    'https://datahub.afyaanalytics.com',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'https://2c97-129-222-187-199.ngrok-free.app',
    'https://8fa2-105-164-5-177.ngrok-free.app'
]

# Add this — CORS was missing the ngrok origin
CORS_ALLOWED_ORIGINS = [
    'https://datahub.afyaanalytics.com',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'https://2c97-129-222-187-199.ngrok-free.app',
    'https://8fa2-105-164-5-177.ngrok-free.app'
]

# Or for dev, just allow everything:
# CORS_ALLOW_ALL_ORIGINS = True
# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    # daphne must be first to override runserver with ASGI
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party
    'rest_framework',
    'rest_framework_simplejwt',
    'drf_spectacular',
    'django_filters',
    'channels',
    # Local apps
    'core',
    'authentication',
    'analytics_app',
    'warehouse',
    'airflow_ui',
    'self_service',
    'agents',
    'corsheaders',
    'catalog',
    'forecasting',
]

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # must be first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'core.middleware.AuditLogMiddleware',
    'core.middleware.RequestLoggingMiddleware',
]

ROOT_URLCONF = 'airflow_dashboard.urls'

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...").strip()

CUBE_API_URL = os.getenv("CUBE_API_URL", "http://localhost:4000").strip()
CUBE_API_TOKEN = os.getenv("CUBE_API_TOKEN", "your-cube-api-secret").strip()
ANALYTICS_TEAM_EMAIL = os.getenv("ANALYTICS_TEAM_EMAIL", "data@afya.ai").strip()

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'core.context_processors.notifications',
                'core.context_processors.brand_settings',
                'core.context_processors.module_access',
            ],
        },
    },
]

WSGI_APPLICATION = 'airflow_dashboard.wsgi.application'
ASGI_APPLICATION = 'airflow_dashboard.asgi.application'
WHAPI_TOKEN = os.getenv("WHAPI_TOKEN", "your-whapi-channel-token").strip()
WHAPI_URL = os.getenv("WHAPI_URL", "https://gate.whapi.cloud").strip()
# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

if DEBUG:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.environ.get('POSTGRES_DBNAME', '').strip(),
            'USER': os.environ.get('POSTGRES_USERNAME', '').strip(),
            'PASSWORD': os.environ.get('POSTGRES_PASSWORD', '').strip(),
            'HOST': os.environ.get('POSTGRES_HOST', 'localhost').strip(),
            'PORT': os.environ.get('POSTGRES_PORT', '5432').strip(),
            'CONN_MAX_AGE': 60,
            'OPTIONS': {
                'connect_timeout': 10,
            },
        }
    }

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Africa/Nairobi'
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static & Media files
# ---------------------------------------------------------------------------

STATIC_URL = 'static/'
STATIC_ROOT = '/usr/src/app/static'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

MEDIA_URL = '/media/'
MEDIA_ROOT = '/usr/src/app/media'

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

LOGIN_URL = '/auth/login/'
LOGIN_REDIRECT_URL = '/analytics/'
LOGOUT_REDIRECT_URL = '/auth/login/'

# ---------------------------------------------------------------------------
# External Services
# ---------------------------------------------------------------------------

GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', '').strip()

STREAMLIT_BASE_URL = os.getenv(
    'STREAMLIT_BASE_URL',
    'http://localhost:8501'
).rstrip('/')

# Browser-facing Redash URL (used to build iframe embed src's — must be
# reachable from the user's browser, not just from inside the docker network).
REDASH_BASE_URL = os.getenv(
    'REDASH_BASE_URL',
    'http://localhost:5050'
).rstrip('/')

# Admin-user API key used server-side to provision Redash Groups/Data
# Sources per facility (see analytics_app/management/commands/provision_redash_facility.py).
REDASH_ADMIN_API_KEY = os.getenv('REDASH_ADMIN_API_KEY', '').strip()

AIRFLOW_BASE_URL = os.getenv('AIRFLOW_BASE_URL', 'http://localhost:8080').rstrip('/')
AIRFLOW_USERNAME = os.getenv('AIRFLOW_USERNAME', 'airflow').strip()
AIRFLOW_PASSWORD = os.getenv('AIRFLOW_PASSWORD', 'airflow').strip()

SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT', '').strip()
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER', '').strip()
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD', '').strip()
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE', '').strip()
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE', '').strip()
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC').strip()

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

EMAIL_BACKEND = os.getenv(
    'EMAIL_BACKEND',
    'django.core.mail.backends.smtp.EmailBackend'
).strip()
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com').strip()
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587').strip())
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'True').lower() in ('true', '1', 'yes')
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '').strip()
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '').strip()
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'noreply@afyaanalytics.com').strip()

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DATETIME_FORMAT': '%Y-%m-%dT%H:%M:%SZ',
}

# ---------------------------------------------------------------------------
# DRF Spectacular (OpenAPI / Swagger)
# ---------------------------------------------------------------------------

SPECTACULAR_SETTINGS = {
    'TITLE': 'Afya DataHub API',
    'DESCRIPTION': 'Healthcare Analytics Platform API — powering data-driven decisions across facilities.',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False,
    'SWAGGER_UI_SETTINGS': {
        'deepLinking': True,
        'persistAuthorization': True,
        'displayOperationId': True,
    },
    'COMPONENT_SPLIT_REQUEST': True,
    'SORT_OPERATIONS': False,
    'TAGS': [
        {'name': 'auth', 'description': 'Authentication endpoints'},
        {'name': 'core', 'description': 'Core platform resources'},
        {'name': 'analytics', 'description': 'Analytics dashboards'},
        {'name': 'warehouse', 'description': 'Data warehouse'},
        {'name': 'pipelines', 'description': 'Airflow pipeline management'},
    ],
}

# ---------------------------------------------------------------------------
# Simple JWT
# ---------------------------------------------------------------------------

from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': False,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
}

# ---------------------------------------------------------------------------
# Messages → Bootstrap alert classes
# ---------------------------------------------------------------------------

MESSAGE_TAGS = {
    messages.DEBUG: 'secondary',
    messages.INFO: 'info',
    messages.SUCCESS: 'success',
    messages.WARNING: 'warning',
    messages.ERROR: 'danger',
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {module}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs/afya_datahub.log',
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'WARNING',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console', 'file'],
            'level': 'ERROR',
            'propagate': False,
        },
        'authentication': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'analytics_app': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'warehouse': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'airflow_ui': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'core': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'self_service': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'agents': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# ---------------------------------------------------------------------------
# Brand / App settings (exposed to templates via context processor)
# ---------------------------------------------------------------------------

AFYA_BRAND = {
    'APP_NAME': 'Afya DataHub',
    'TAGLINE': 'Empowering Healthcare with Data',
    'COLORS': {
        'blue': '#0072CE',
        'teal': '#0BB99F',
        'cool_blue': '#003467',
        'orange': '#f5a623',
        'amber': '#D97706',
    },
    'SUPPORT_EMAIL': 'data@afya.ai',
    'VERSION': '2.0.0',
}

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

SESSION_COOKIE_AGE = 86400  # 24 hours
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# ---------------------------------------------------------------------------
# Django Channels — Self-Service Analytics WebSocket
# ---------------------------------------------------------------------------

_REDIS_URL = os.getenv('REDIS_URL', '').strip()

if _REDIS_URL:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {'hosts': [_REDIS_URL]},
        }
    }
else:
    # In-memory layer: suitable for single-process dev; not shared across workers
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
        }
    }

# ---------------------------------------------------------------------------
# Celery — background tasks (agents/tasks.py: Agent Configuration's
# "Generate Missing Metrics" / "Rebuild Embeddings" buttons — both make
# several slow LLM/embedding calls, too slow to run in a request/response
# cycle). Broker/backend default to the `redis` service already defined in
# docker-compose.yaml / docker-compose.dev.yaml, on the same `dashboard-net`
# network as `web` — no new infrastructure, override via env if deployed
# differently.
# ---------------------------------------------------------------------------

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', CELERY_BROKER_URL)
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
CELERY_RESULT_EXPIRES = 60 * 60 * 24  # 24 hours — enough to check a result the next morning

# Runs a task synchronously in-process (no broker/worker needed) when set —
# for local dev without docker-compose's redis service running. Off by
# default so production always actually queues instead of silently blocking.
CELERY_TASK_ALWAYS_EAGER = os.getenv('CELERY_TASK_ALWAYS_EAGER', 'false').strip().lower() == 'true'
