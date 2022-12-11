import streamlit as st
st.markdown("# Gradient Descent")

with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.selectbox('Select Page',['GradientDescent01','GradientDescent02']) 

if app_mode=='GradientDescent01':
    st.title("GradientDescent01") 
    from pages.GradientDescent1.Gradientdescent01 import *
    from matplotlib.animation import FuncAnimation
    import streamlit.components.v1 as components


    
    def f(x):
        return x**2 - 5*np.sin(x)

    def df(x):
        return 2*x - 5*np.cos(x)

    fig = plt.figure(figsize=[10, 7])
    ax = plt.axes(xlim=(-6, 6), ylim=(-10, 60))
    ax.text(-6, 55, 'Hàm số $y=x^2-5\sin{x}$, $step\_multiplier=0.2$, $precision=0.00001$, $start=-10$', fontsize=12)
    label_1 = ax.text(-6, 50, '', fontsize=12)
    label_2 = ax.text(0, 30, '', fontsize=20)

    line, = ax.plot([], [], 'ro-', lw=5)
    x = np.linspace(start=-8, stop=8, num=100)
    y = f(x)
    ax.plot(x,y)

    x_1 = -10
    x_0 = 0
    step_multiplier = 0.2
    precision = 0.00001

    def animate(i):
        global x_0, x_1
        step_size = abs(x_1 - x_0)
        
        if step_size > precision:
            x_0 = x_1
            gradient = df(x_0)
            x_1 = x_0 - step_multiplier * gradient
            x = [x_0, x_1]
            y = [f(x_0), f(x_1)]
            line.set_data(x, y)
            label_2.set_text(str(i))
        label_1.set_text('Lần: ' + str(i) + '/50, cost:' + str(f(x_1)) + ', slope:' + str(df(x_1)))
        return line, 

    anim = FuncAnimation(fig, animate, frames=50, interval=400, blit=True)
    components.html(anim.to_jshtml(), height=1000, width=900)
    

elif app_mode == 'GradientDescent02':
    st.title('GradientDescent02')
    from pages.GradientDescent1.Gradientdescent02 import *
    X = np.random.rand(1000)
    y = 4 + 3 * X + .5*np.random.randn(1000)
    fig, ax = plt.subplots()
    plt.plot(X,y,'bo', markersize = 2)
    # chuyển mảng một chiều thành ma trận
    X = np.array([X])
    y = np.array([y])
    # chuyển vị ma trận
    X=X.T
    y=y.T
    model = LinearRegression()
    model.fit(X, y)
    w0 = model.intercept_
    w1 = model.coef_[0]
    x0 = 0
    y0 = w1*x0 +w0
    x1 = 1
    y1 = w1*x1 +w0
    plt.plot([x0,x1],[y0,y1], 'r')
    st.pyplot(fig)