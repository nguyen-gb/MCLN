import streamlit as st

with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
    
st.markdown("# KNN")

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

    

app_mode = st.selectbox('Select Page',['KNN01','KNN01a','KNN01b']) 

if app_mode=='KNN01':
    st.title("KNN01") 
    from pages.KNN1.KNN01 import *
    fig, ax = plt.subplots()
    plt.plot(nhom0[:,0],nhom0[:,1],'go')
    plt.plot(nhom1[:,0],nhom1[:,1], 'ro')
    plt.plot(nhom2[:,0],nhom2[:,1], 'bo')
    plt.legend([0,1,2])
    st.pyplot(fig)

    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=1)

    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Độ chính xác: %.0f%%' % (accuracy*100))

elif app_mode == 'KNN01a':
    st.title('KNN01a')
    from pages.KNN1.KNN01a import *
    fig, ax = plt.subplots()
    plt.plot(nhom0[:,0],nhom0[:,1],'go')
    plt.plot(nhom1[:,0],nhom1[:,1], 'ro')
    plt.plot(nhom2[:,0],nhom2[:,1], 'bo')
    plt.legend([0,1,2])
    st.pyplot(fig)
    res = train_test_split(data, labels, 
                        train_size=0.8,
                        test_size=0.2,
                        random_state=1)

    train_data, test_data, train_labels, test_labels = res 
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    accuracy = accuracy_score(predicted, test_labels)
    st.write('Độ chính xác: %.0f%%' % (accuracy*100))
    joblib.dump(knn, "knn.pkl")

elif app_mode == 'KNN01b':
    st.title('Nhan dang chu so viet tay')
    import numpy as np
    import joblib
    import tensorflow as tf
    from PIL import ImageTk, Image
    import cv2
    from streamlit_drawable_canvas import st_canvas


    def btn_create_digit_click():
        index = np.random.randint(0,9999, 100)
        digit = np.zeros((28*10, 28*10), np.uint8)
        k =  0
        for x in range(0, 10):
            for y in range(0, 10):
                digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
                k = k+1
        cv2.imwrite('digit.jpg', digit)
        image = Image.open('digit.jpg')
        img = image.resize((280,280), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(img)
        cvs_digit.create_image(0, 0, image = image_tk)
        
    def btn_recognition_click():
        digit_data = np.zeros((100, 28, 28), np.uint8)

        for i in range(0, 100):
            digit_data[i] = X_test[index[i]]

        RESHAPED = 784 # 28*28
        digit_data = digit_data.reshape(100, RESHAPED)
        predicted = knn.predict(digit_data)
        k = 0
        ket_qua = ''
        for x in range(0, 10):
            for y in range(0, 10):
                ket_qua = ket_qua + '%3d' % predicted[k]
                k = k+1
            ket_qua = ket_qua  + '\n'
        ket_qua = ket_qua[:-1]

    knn = joblib.load('knn_digit.pkl')
    index = None
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    



    cvs_digit = st_canvas()
    btn_create_digit = st.button('Create Digit', on_click= btn_create_digit_click)
    btn_recognition = st.button('Recognition', on_click= btn_recognition_click)   

