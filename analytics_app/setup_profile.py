from accounts.models import UserProfile
from django.contrib.auth.models import User

for user in User.objects.all():
    profile, created = UserProfile.objects.get_or_create(user=user)

    if created:
        print(f"Created profile for {user.username}")
    else:
        print(f"Profile already exists for {user.username}")