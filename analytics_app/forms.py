"""
Analytics app forms.
"""

from django import forms
from .models import Dashboard


class DashboardForm(forms.ModelForm):
    """Full create/update form for a Dashboard record."""

    class Meta:
        model = Dashboard
        fields = [
            'name', 'description', 'category',
            'client', 'facility',
            'streamlit_url',
            'redash_query_id', 'redash_visualization_id', 'redash_api_key',
            'redash_dashboard_url',
            'thumbnail',
            'is_active', 'is_public', 'order',
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Dashboard name',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional description…',
            }),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'client': forms.Select(attrs={'class': 'form-select'}),
            'facility': forms.Select(attrs={'class': 'form-select'}),
            'streamlit_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://… (leave blank if using a Redash embed below)',
            }),
            'redash_query_id': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. 42',
            }),
            'redash_visualization_id': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. 108',
            }),
            'redash_api_key': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Query-level API key from Redash',
            }),
            'redash_dashboard_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://.../public/dashboards/<token>?org_slug=default '
                                '(leave blank if using a single query/visualization above)',
            }),
            'thumbnail': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'order': forms.NumberInput(attrs={'class': 'form-control', 'min': 0}),
        }


class ReportingQueryForm(forms.Form):
    """Superuser form for submitting a custom SQL query to be created in Redash."""

    name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Query name'}),
    )
    sql_text = forms.CharField(
        label='SQL',
        widget=forms.Textarea(attrs={
            'class': 'form-control', 'rows': 10,
            'placeholder': 'SELECT * FROM HOSPITALS.REPORTING.SOME_TABLE',
        }),
    )
    data_source_id = forms.ChoiceField(widget=forms.Select(attrs={'class': 'form-select'}))

    def __init__(self, *args, data_source_choices=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['data_source_id'].choices = data_source_choices


class DashboardSearchForm(forms.Form):
    """Lightweight search/filter form for the dashboard list view."""

    q = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search dashboards…',
        }),
    )
    category = forms.ChoiceField(
        required=False,
        choices=[('', 'All Categories')] + Dashboard.CATEGORY_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
    )
