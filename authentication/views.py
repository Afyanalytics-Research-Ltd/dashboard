# accounts/views.py

from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm
from django.contrib.auth.models import User, Group
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse
from django.conf import settings

from .models import UserProfile
from authentication.roles import DEFAULT_ROLE


# --- LOGIN ------------------------------------------------------------

def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    form = AuthenticationForm(request, data=request.POST or None)

    if request.method == "POST" and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect("dashboard")

    return render(request, "accounts/login.html", {"form": form})


# --- LOGOUT -----------------------------------------------------------

def logout_view(request):
    logout(request)
    return redirect("login")


# --- SIGNUP -----------------------------------------------------------

def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password")
        confirm = request.POST.get("confirm_password")

        phone_number = request.POST.get("phone_number", "").strip()
        client = request.POST.get("client", "").strip()
        facility = request.POST.get("facility", "").strip()

        # --- Validation ---
        if not username or not email or not password:
            messages.error(request, "All required fields must be filled")
            return redirect("signup")

        if password != confirm:
            messages.error(request, "Passwords do not match")
            return redirect("signup")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("signup")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already in use")
            return redirect("signup")

        # --- Create user ---
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        # --- Save profile data ---
        profile = user.profile
        profile.phone_number = phone_number
        profile.client = client
        profile.facility = facility
        profile.save()

        # --- Assign default role ---
        group = Group.objects.get(name=DEFAULT_ROLE)
        user.groups.add(group)

        # --- Login ---
        login(request, user)

        messages.success(request, "Account created successfully")

        return redirect("dashboard")

    return render(request, "accounts/signup.html")


# --- PROFILE ----------------------------------------------------------

def profile_view(request):
    profile = request.user.profile

    if request.method == "POST":
        profile.phone_number = request.POST.get("phone_number")
        profile.client = request.POST.get("client")
        profile.facility = request.POST.get("facility")
        profile.job_title = request.POST.get("job_title")
        profile.save()

        messages.success(request, "Profile updated successfully")
        return redirect("profile")

    return render(request, "accounts/profile.html", {"profile": profile})


# --- PASSWORD RESET ---------------------------------------------------

def password_reset_view(request):
    form = PasswordResetForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        form.save(
            request=request,
            use_https=request.is_secure(),
            email_template_name="accounts/password_reset_email.html",
        )
        messages.success(request, "Password reset email sent")
        return redirect("login")

    return render(request, "accounts/password_reset.html", {"form": form})