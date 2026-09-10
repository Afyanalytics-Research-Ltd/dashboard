from django import forms


class ScheduledBrowserTaskForm(forms.Form):
    """Name + instructions + a simple daily HH:MM picker.

    Deliberately narrower than full cron — "every day at 4pm" is the
    request this was built for. The CrontabSchedule this creates always has
    day_of_week/day_of_month/month_of_year='*', only hour/minute set.
    """

    name = forms.CharField(
        max_length=120,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Daily price check"}),
    )
    instructions = forms.CharField(
        widget=forms.Textarea(attrs={
            "class": "form-control font-monospace",
            "rows": 6,
            "placeholder": "go to https://example.com/pricing\nextract: the current plan prices",
        }),
        help_text="One instruction per line, run in order on a single browser session each run.",
    )
    hour = forms.IntegerField(min_value=0, max_value=23, widget=forms.NumberInput(attrs={"class": "form-control"}))
    minute = forms.IntegerField(min_value=0, max_value=59, widget=forms.NumberInput(attrs={"class": "form-control"}))
    is_active = forms.BooleanField(required=False, initial=True, widget=forms.CheckboxInput(attrs={"class": "form-check-input"}))

    def clean_instructions(self):
        lines = [l.strip() for l in self.cleaned_data["instructions"].splitlines() if l.strip()]
        if not lines:
            raise forms.ValidationError("Enter at least one instruction.")
        return "\n".join(lines)
