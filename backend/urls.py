"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core.views import InsertUserData, GetUserData, SignUp, EditUserData, DeleteUserData, Analysis, index, upload

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name = 'index'),
    path('upload/', upload, name = 'upload'),
    path('insert/',InsertUserData.as_view(), name = 'insert'),
    path('get/',GetUserData.as_view(), name = 'get'),
    path('signup/', SignUp.as_view(), name = 'signup'),
    path('edit/', EditUserData.as_view(), name = 'edit'),
    path('delete/', DeleteUserData.as_view(), name = 'delete'),
    path('analysis/', Analysis.as_view(), name = 'analysis'),
    path('rest-auth/', include('rest_auth.urls')),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
