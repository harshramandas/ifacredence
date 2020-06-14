
from rest_framework.decorators import api_view, permission_classes
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import UserSerializer, UserDataSerializerGet, UserDataSerializerInsert
from .models import UserData, User, NativeFile
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage

@api_view(['GET'])
@permission_classes([AllowAny])
def index(request):
    ob = NativeFile.objects.all()
    my_dict = {"files" : ob }
    return render(request, 'index.html', context = my_dict)

@api_view(['POST'])
@permission_classes([AllowAny])
def upload(request):
    uploaded_file = request.FILES['document']
    fs = FileSystemStorage()
    name = fs.save(uploaded_file.name, uploaded_file)
    url = fs.url(name)



class SignUp(APIView):
    permission_classes = (AllowAny,)
    def post(self, request, *args, **kwargs):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            User.objects.create_user(
                username = request.data["email"],
                email=request.data["email"],
                password=request.data["password"],
                first_name=request.data["first_name"],
                last_name=request.data["last_name"]
            )
            return Response(serializer.data)
        return Response(serializer.errors)



class InsertUserData(APIView):
    permission_classes = (IsAuthenticated,)
    def post(self, request, *args, **kwargs):
        serializer = UserDataSerializerInsert(data=request.data)
        # print(serializer.data, type(serializer.data))
        if serializer.is_valid():
            ob = UserData.objects.create(
                name = request.data["name"],
                amount=request.data["amount"],
                cashflow=request.data["cashflow"],
                category=request.data["category"],
                interval=request.data["interval"],
                description=request.data["description"],
                owner=request.user
            )
            return Response(ob.pk)
        return Response(serializer.errors)



class DeleteUserData(APIView):
    permission_classes = (IsAuthenticated,)
    def post(self, request, *args, **kwargs):
        UserData.objects.filter(id=request.data["id"]).delete()
        return Response({"deletedpk":request.data["id"]})



class EditUserData(APIView):
    permission_classes = (IsAuthenticated,)
    def post(self, request, *args, **kwargs):
        serializer = UserDataSerializerGet(data=request.data)
        # print(serializer.data, type(serializer.data))
        if serializer.is_valid():
            UserData.objects.filter(id=request.data["id"]).update(
                name = request.data["name"],
                amount=request.data["amount"],
                cashflow=request.data["cashflow"],
                category=request.data["category"],
                interval=request.data["interval"],
                description=request.data["description"]
            )
            ob = UserData.objects.get(pk=request.data["id"])
            return Response(ob.pk)
        return Response(serializer.errors)



class GetUserData(APIView):
    permission_classes = (IsAuthenticated,)
    def get(self, request, *args, **kwargs):
        qs = UserData.objects.filter(owner=request.user)
        serializer = UserDataSerializerGet(qs, many = True)
        # print(serializer.data)
        return Response(serializer.data)



class Analysis(APIView):
    permission_classes = (IsAuthenticated,)
    def post(self, request, *args, **kwargs):
        import yfinance as yf
        import datetime
        ticker = request.data['ticker']
        tickerData = yf.Ticker(ticker)
        import twitter

        api=twitter.Api(consumer_key='XUjMfGNoBHciRVC4kNCOa3Esw', 
                        consumer_secret='M5qyE97XarqyXfi0EE9YhCIx7rM7SqhhLKSd2zycPUx3d7WulD', 
                        access_token_key='1170978534170361857-YILY0jBYMVhSPi7oGdwIlfKvghp2GO', 
                        access_token_secret='rMuhIXdwOiK4bYiLnpfG6jBY7z8zTqjvTE9hcbabHMNk2')
        print(api.VerifyCredentials())

        twitter_data_raw=[]

        def create_test_data(search_string):
            try:
                tweets_fetched=api.GetSearch(search_string, count=2000)
                print("fetched")
                for status in tweets_fetched:
                    if status.retweet_count>0:
                        if status.text not in twitter_data_raw:
                            twitter_data_raw.append(status.text)
                    else:
                        twitter_data_raw.append(status.text)
                    if len(twitter_data_raw)==100:
                        break
            except:
                print("sorry")
                return None
                        
        create_test_data(tickerData.info['shortName'])

        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer

        twitter_data=[]
        remove=['at_user','url']

        for tweet in twitter_data_raw:
            review=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
            review=re.sub('@[^\s]+','at_user',review)
            review=re.sub(r'#([^\s]+)', r'\1', review)
            review=re.sub('[^a-zA-Z]','  ',review)
            review=review.lower()
            review=review.split()
            ps=PorterStemmer()
            review=[ps.stem(word) for word in review if word not in set(stopwords.words('english')) and word not in remove]
            review=' '.join(review)
            twitter_data.append(review)

        pol = 0    
        result=[]
        from textblob import TextBlob
        for check_tweet in twitter_data:
            analysis=TextBlob(check_tweet)
            if analysis.sentiment.polarity>0:
                res='positive'
                pol += analysis.sentiment.polarity
            elif analysis.sentiment.polarity==0:
                res='neutral'
                pol += analysis.sentiment.polarity
            else:
                res='negative'
                pol += analysis.sentiment.polarity
            result.append(res)

        pos=result.count('positive')
        neg=result.count('negative')
        neu=result.count('neutral')
        sen = 0
        tim = 0
        import numpy as np
        import pandas as pd
        # import tensorflow as tf
        # import tensorflow.keras as keras
        # from tensorflow.keras.models import Sequential
        # from tensorflow.keras.layers import Layer, Dense, Activation, Embedding, Flatten, LeakyReLU, PReLU, ELU, BatchNormalization, Dropout
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(os.path.join(settings.BASE_DIR, 'data.csv'), index_col=0)
        df = df.dropna()
        df.rename(index=df.Date,inplace=True)
        df = df.drop(columns=['Date','Change'])
        prev = 91.53
        Change = []
        for i in df.Close:
            Change.append((i - prev)/i)
            prev = i
        df['Change'] = Change
        X_train, X_test, y_train, y_test = train_test_split(
                                                            df.Polarity,
                                                            df.Change,
                                                            test_size=0.25,
                                                            random_state=42)
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(np.array(X_train).reshape(-1,1), y_train)
        
        sen = regressor.predict(np.array(pol).reshape(1,-1))[0]
        # day = datetime.date.today()
        # tickerDf = tickerData.history(period='1d', start=str(day-datetime.timedelta(days=700)), end=str(day))
        # time = tickerDf.index
        # series = tickerDf.Close.values
        # def window_dataset(series, window_size, batch_size=32, shuffle_buffer=200):
        #     dataset = tf.data.Dataset.from_tensor_slices(series)
        #     dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        #     dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        #     dataset = dataset.shuffle(shuffle_buffer)
        #     dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        #     dataset = dataset.batch(batch_size).prefetch(1)
        #     return dataset
        # def seq2seq_window_dataset(series, window_size, batch_size=32, shuffle_buffer=200):
        #     series = tf.expand_dims(series, axis=-1)
        #     ds = tf.data.Dataset.from_tensor_slices(series)
        #     ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        #     ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        #     ds = ds.shuffle(shuffle_buffer)
        #     ds = ds.map(lambda w: (w[:-1], w[1:]))
        #     return ds.batch(batch_size).prefetch(1)
        # keras.backend.clear_session()
        # tf.random.set_seed(42)
        # np.random.seed(42)

        # window_size = 30
        # train_set = seq2seq_window_dataset(X_train, window_size,
        #                                 batch_size=128)
        # valid_set = seq2seq_window_dataset(X_test, window_size,
        #                                 batch_size=128)

        # model = keras.models.Sequential([
        # keras.layers.Conv1D(filters=32, kernel_size=5,
        #                     strides=1, padding="causal",
        #                     activation="relu",
        #                     input_shape=[None, 1]),
        # keras.layers.LSTM(32, return_sequences=True),
        # keras.layers.LSTM(32, return_sequences=True),
        # keras.layers.Dense(1),
        # keras.layers.Lambda(lambda x: x * 200)
        # ])
        # optimizer = keras.optimizers.SGD(lr=5 * 1e-5, momentum=0.9)
        # model.compile(loss=keras.losses.Huber(),
        #             optimizer=optimizer,
        #             metrics=["mae"])

        # model_checkpoint = keras.callbacks.ModelCheckpoint(
        #     "my_checkpoint.h5", save_best_only=True)
        # early_stopping = keras.callbacks.EarlyStopping(patience=50)
        # model.fit(train_set, epochs=500,
        #         validation_data=valid_set,
        #         callbacks=[early_stopping, model_checkpoint])
        # model = keras.models.load_model(os.path.join(settings.BASE_DIR, "my_checkpoint.h5"))
        # def model_forecast(model, series, window_size):
        #     ds = tf.data.Dataset.from_tensor_slices(series)
        #     ds = ds.window(window_size, shift=1, drop_remainder=True)
        #     ds = ds.flat_map(lambda w: w.batch(window_size))
        #     ds = ds.batch(32).prefetch(1)
        #     forecast = model.predict(ds)
        #     return forecast
        # rnn_forecast = model_forecast(model, series[:,  np.newaxis], 30)
        # rnn_forecast = rnn_forecast[-1, -1, 0]
        # tim = rnn_forecast
        return Response(data={"pos" : pos, "neg" : neg, "neu" : neu, "sen": sen})

            