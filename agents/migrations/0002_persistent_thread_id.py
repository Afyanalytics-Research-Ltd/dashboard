from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agents', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='whatsappchatstate',
            old_name='last_metric_thread_id',
            new_name='thread_id',
        ),
        migrations.CreateModel(
            name='ConversationThread',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(db_index=True, max_length=255, unique=True)),
                ('thread_id', models.CharField(max_length=64)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
