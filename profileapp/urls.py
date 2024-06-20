from django.urls import path
from . import views

urlpatterns = [
    path('', views.SignupPage , name="signup"),
    path('base',views.base,name='base'),
    path('login/',views.LoginPage,name='login'),
    
    path('logout/',views.LogoutPage,name='logout'),
    path('facebook',views.facebook,name='facebook'),
    
    path('singleuser', views.singleuser, name='singleuser'),
    path('blog', views.blog, name='blog'),
    path('followers', views.followers, name='followers'),
    path('urduuser', views.urduuser, name='urduuser'),
    path('urdufollowings', views.urdufollowings, name='urdufollowings'),
    path('save_to_db',views.save_to_db,name='save_to_db'),
    path('history', views.history,name='history'),
    path('classify_tweet/', views.classifyTweet, name='classify_tweet')
   
]