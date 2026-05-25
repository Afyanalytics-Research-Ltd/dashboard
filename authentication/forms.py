"""
Authentication forms for Afya DataHub.

All form widgets carry Bootstrap 5 classes so templates only need to render
{{ form.field }} without extra HTML gymnastics.
"""

import logging

from django import forms
from django.contrib.auth import get_user_model, password_validation
from django.contrib.auth.forms import (
    AuthenticationForm,
    PasswordChangeForm as DjangoPasswordChangeForm,
    UserCreationForm,
)
from django.core.exceptions import ValidationError

from .models import UserProfile

logger = logging.getLogger(__name__)

User = get_user_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CTRL = 'form-control'
_CTRL_LG = 'form-control form-control-lg'
_CHECK = 'form-check-input'
_SELECT = 'form-select'
_TEXTAREA = 'form-control'


def _widget(tag, cls=_CTRL, **attrs):
    """Return an appropriately classed Django form widget."""
    mapping = {
        'text': forms.TextInput,
        'email': forms.EmailInput,
        'password': forms.PasswordInput,
        'tel': forms.TextInput,
        'file': forms.ClearableFileInput,
        'textarea': forms.Textarea,
        'select': forms.Select,
        'checkbox': forms.CheckboxInput,
        'number': forms.NumberInput,
    }
    klass = mapping.get(tag, forms.TextInput)
    merged = {'class': cls}
    merged.update(attrs)
    return klass(attrs=merged)


# ---------------------------------------------------------------------------
# LoginForm
# ---------------------------------------------------------------------------

class LoginForm(AuthenticationForm):
    """
    Extends Django's AuthenticationForm with Bootstrap 5 styling.
    Accepts both username and e-mail in the username field.
    """

    username = forms.CharField(
        label='Username or Email',
        widget=forms.TextInput(attrs={
            'class': _CTRL_LG,
            'placeholder': 'Enter your username or email',
            'autofocus': True,
            'autocomplete': 'username',
        }),
    )
    password = forms.CharField(
        label='Password',
        strip=False,
        widget=forms.PasswordInput(attrs={
            'class': _CTRL_LG,
            'placeholder': 'Enter your password',
            'autocomplete': 'current-password',
            'id': 'id_password',
        }),
    )

    def confirm_login_allowed(self, user):
        super().confirm_login_allowed(user)
        if not user.is_active:
            raise ValidationError('This account has been deactivated.', code='inactive')


# ---------------------------------------------------------------------------
# SignupForm
# ---------------------------------------------------------------------------

class SignupForm(UserCreationForm):
    """
    Registration form. Creates the User object; the profile is updated in
    the view (or by the post_save signal).
    """

    first_name = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': 'First name'}),
    )
    last_name = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': 'Last name'}),
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': _CTRL, 'placeholder': 'you@example.com'}),
    )
    phone_number = forms.CharField(
        required=False,
        max_length=32,
        widget=forms.TextInput(attrs={
            'class': _CTRL,
            'placeholder': '+254 700 000 000',
            'id': 'id_phone_number',
        }),
    )
    job_title = forms.CharField(
        required=False,
        max_length=120,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': 'e.g. County Health Officer'}),
    )
    terms = forms.BooleanField(
        required=True,
        label='I agree to the Terms of Service and Privacy Policy',
        widget=forms.CheckboxInput(attrs={'class': _CHECK}),
        error_messages={'required': 'You must accept the terms and conditions.'},
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'phone_number', 'job_title',
                  'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply Bootstrap classes to inherited fields
        self.fields['username'].widget.attrs.update({
            'class': _CTRL,
            'placeholder': 'Choose a username',
            'autocomplete': 'username',
        })
        self.fields['password1'].widget.attrs.update({
            'class': _CTRL,
            'placeholder': 'Create a strong password',
            'autocomplete': 'new-password',
            'id': 'id_password1',
        })
        self.fields['password2'].widget.attrs.update({
            'class': _CTRL,
            'placeholder': 'Repeat your password',
            'autocomplete': 'new-password',
            'id': 'id_password2',
        })
        self.fields['username'].help_text = (
            'Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.'
        )

    def clean_email(self):
        email = self.cleaned_data.get('email', '').lower()
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError('An account with this email already exists.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name', '')
        user.last_name = self.cleaned_data.get('last_name', '')
        user.email = self.cleaned_data.get('email', '')
        if commit:
            user.save()
        return user


# ---------------------------------------------------------------------------
# ProfileUpdateForm
# ---------------------------------------------------------------------------

class ProfileUpdateForm(forms.Form):
    """
    Combined User + UserProfile update form.
    This is a plain Form (not ModelForm) because it straddles two models.
    The view is responsible for saving to both.
    """

    first_name = forms.CharField(
        max_length=150,
        required=False,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': 'First name'}),
    )
    last_name = forms.CharField(
        max_length=150,
        required=False,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': 'Last name'}),
    )
    email = forms.EmailField(
        required=False,
        widget=forms.EmailInput(attrs={'class': _CTRL, 'placeholder': 'you@example.com'}),
    )
    phone_number = forms.CharField(
        required=False,
        max_length=32,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': '+254 700 000 000'}),
    )
    job_title = forms.CharField(
        required=False,
        max_length=120,
        widget=forms.TextInput(attrs={'class': _CTRL, 'placeholder': 'e.g. County Health Officer'}),
    )
    bio = forms.CharField(
        required=False,
        max_length=500,
        widget=forms.Textarea(attrs={
            'class': _TEXTAREA,
            'rows': 4,
            'placeholder': 'Tell us a little about yourself…',
        }),
    )
    avatar = forms.ImageField(
        required=False,
        widget=forms.ClearableFileInput(attrs={'class': 'form-control form-control-sm'}),
    )

    def __init__(self, user=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._user = user
        if user is not None:
            self.fields['first_name'].initial = user.first_name
            self.fields['last_name'].initial = user.last_name
            self.fields['email'].initial = user.email
            if hasattr(user, 'profile'):
                p = user.profile
                self.fields['phone_number'].initial = p.phone_number
                self.fields['job_title'].initial = p.job_title
                self.fields['bio'].initial = p.bio

    def clean_email(self):
        email = self.cleaned_data.get('email', '').lower()
        if not email:
            return email
        qs = User.objects.filter(email__iexact=email)
        if self._user:
            qs = qs.exclude(pk=self._user.pk)
        if qs.exists():
            raise ValidationError('This email address is already in use.')
        return email

    def clean_avatar(self):
        avatar = self.cleaned_data.get('avatar')
        if avatar:
            max_size = 5 * 1024 * 1024  # 5 MB
            if avatar.size > max_size:
                raise ValidationError('Avatar file must be smaller than 5 MB.')
            allowed = ('image/jpeg', 'image/png', 'image/gif', 'image/webp')
            ct = getattr(avatar, 'content_type', '')
            if ct and ct not in allowed:
                raise ValidationError('Only JPEG, PNG, GIF or WebP images are allowed.')
        return avatar

    def save(self):
        """Persist changes to both User and UserProfile. Returns the user."""
        if not self._user:
            raise ValueError('ProfileUpdateForm requires a user instance.')
        user = self._user
        d = self.cleaned_data
        user.first_name = d.get('first_name', '')
        user.last_name = d.get('last_name', '')
        if d.get('email'):
            user.email = d['email']
        user.save(update_fields=['first_name', 'last_name', 'email'])

        profile = user.profile
        profile.phone_number = d.get('phone_number', '')
        profile.job_title = d.get('job_title', '')
        profile.bio = d.get('bio', '')
        if d.get('avatar'):
            profile.avatar = d['avatar']
        profile.save(update_fields=['phone_number', 'job_title', 'bio', 'avatar', 'updated_at'])
        return user


# ---------------------------------------------------------------------------
# PasswordChangeForm
# ---------------------------------------------------------------------------

class PasswordChangeForm(DjangoPasswordChangeForm):
    """
    Extends Django's PasswordChangeForm with Bootstrap 5 styling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['old_password'].widget.attrs.update({
            'class': _CTRL,
            'placeholder': 'Current password',
            'autocomplete': 'current-password',
            'id': 'id_old_password',
        })
        self.fields['new_password1'].widget.attrs.update({
            'class': _CTRL,
            'placeholder': 'New password',
            'autocomplete': 'new-password',
            'id': 'id_new_password1',
        })
        self.fields['new_password2'].widget.attrs.update({
            'class': _CTRL,
            'placeholder': 'Confirm new password',
            'autocomplete': 'new-password',
            'id': 'id_new_password2',
        })


# ---------------------------------------------------------------------------
# AvatarUploadForm
# ---------------------------------------------------------------------------

class AvatarUploadForm(forms.Form):
    """Standalone avatar upload with strict validation."""

    avatar = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/gif,image/webp',
        }),
        help_text='JPEG, PNG, GIF or WebP. Max 5 MB.',
    )

    def clean_avatar(self):
        avatar = self.cleaned_data.get('avatar')
        if avatar:
            max_size = 5 * 1024 * 1024
            if avatar.size > max_size:
                raise ValidationError('File size must not exceed 5 MB.')
            allowed = ('image/jpeg', 'image/png', 'image/gif', 'image/webp')
            ct = getattr(avatar, 'content_type', '')
            if ct and ct not in allowed:
                raise ValidationError('Unsupported image format.')
        return avatar
