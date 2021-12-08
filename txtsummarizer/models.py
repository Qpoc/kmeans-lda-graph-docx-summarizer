from django.db import models

# Create your models here.
class News(models.Model):
    news_title = models.CharField(max_length=200)
    news_content = models.TextField()
    