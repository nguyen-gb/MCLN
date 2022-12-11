from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error


with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

st.header('Regression')
st.subheader('Bai01')
st.markdown('**Height (cm), input data, each row is a data point**')
# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

st.write('Height: ',X.T)
st.write('Weight:',y)

regr = linear_model.LinearRegression()
regr.fit(X, y) # in scikit-learn, each sample is one row
# Compare two results
st.markdown('<p style="color:green; font-size:20px;">Compare two results !!</p>', unsafe_allow_html=True)
st.write("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)

X = X[:,0]

fig,ax = plt.subplots()
plt.plot(X, y, 'ro')
a = regr.coef_[0]
b = regr.intercept_
x1 = X[0]
y1 = a*x1 + b
x2 = X[12]
y2 = a*x2 + b
x = [x1, x2]
y = [y1, y2]

plt.plot(x, y)
st.header('Linear Regression Scatter chart')
st.pyplot(fig)


st.header('Regression')
st.subheader('Bai02')
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
X2 = X**2
st.write('X:',X.T) 
st.write('X2:',X2.T) 
X_poly = np.hstack((X, X2))
st.write('X_poly:',X_poly) 

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
st.write('lin_reg.coef:',lin_reg.coef_)
st.write('lin_reg.intercept:',lin_reg.intercept_)
a = lin_reg.intercept_[0]
b = lin_reg.coef_[0,0]
c = lin_reg.coef_[0,1]
st.write('intercept_[0]:',a)
st.write('coef_[0,0]:',b)
st.write('coef_[0,1]:',c)

x_ve = np.linspace(-3,3,m)
y_ve = a + b*x_ve + c*x_ve**2

fig, ax = plt.subplots()
plt.plot(X, y, 'o')
plt.plot(x_ve, y_ve, 'r')

#Tinh sai so
st.markdown('<p style="color:green; font-size:20px;">Tính sai số</p>', unsafe_allow_html=True)
loss = 0 
for i in range(0, m):
    y_mu = a + b*X_poly[i,0] + c*X_poly[i,1]
    sai_so = (y[i] - y_mu)**2 
    loss = loss + sai_so
loss = loss/(2*m)
st.write('loss = %.6f' % loss)

# Tinh sai so cua scikit-learn
st.markdown('<p style="color:blue; font-size:20px;">Tính sai số của scikit-learn</p>', unsafe_allow_html=True)

y_train_predict = lin_reg.predict(X_poly)
st.write('y_train_predict:', y_train_predict)
sai_so_binh_phuong_trung_binh = mean_squared_error(y, y_train_predict)
st.write('Sai số bình phương trung bình: %.6f' % (sai_so_binh_phuong_trung_binh/2))
plt.show()

st.header('Linear Regression Scatter chart')
st.pyplot(fig)


st.header('Regression')
st.subheader('Bai03')
st.markdown('**Height (cm), input data, each row is a data point**')
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

regr = linear_model.LinearRegression()
temp = regr.fit(X, y) # in scikit-learn, each sample is one row

st.write('Height: ',X.T)
st.write('Weight:',y)

st.markdown('<p style="color:green; font-size:20px;">Compare two results !!</p>', unsafe_allow_html=True)

st.write("scikit-learn’s solution : w_1 = ", regr.coef_[0], "w_0 = ", regr.intercept_)

X = X[:,0]
fig, ax = plt.subplots()
plt.plot(X, y, 'ro')

a = regr.coef_[0]
b = regr.intercept_
x1 = X[0]
y1 = a*x1 + b
x2 = X[12]
y2 = a*x2 + b
x = [x1, x2]
y = [y1, y2]

plt.plot(x, y)

st.header('Linear Regression Scatter chart')
st.pyplot(fig)


st.header('Regression')
st.subheader('Bai04 HuberRegressor')
st.markdown('**Height (cm), input data, each row is a data point**')
# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([ 49, 50, 90, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

st.write('Height: ',X.T)
st.write('Weight:',y)

huber_reg = linear_model.HuberRegressor()
huber_reg.fit(X, y) # in scikit-learn, each sample is one row
# Compare two results
st.markdown('<p style="color:green; font-size:20px;">Compare two results !!</p>', unsafe_allow_html=True)
st.write("scikit-learn’s solution : w_1 = ", huber_reg.coef_[0], "w_0 = ", huber_reg.intercept_)

X = X[:,0]
fig,ax = plt.subplots()
plt.plot(X, y, 'ro')
a = huber_reg.coef_[0]
b = huber_reg.intercept_
x1 = X[0]
y1 = a*x1 + b
x2 = X[12]
y2 = a*x2 + b
x = [x1, x2]
y = [y1, y2]


plt.plot(x, y)
st.header('Linear Regression Scatter chart')
st.pyplot(fig)
