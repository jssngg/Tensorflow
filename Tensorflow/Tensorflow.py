import tensorflow as tf
#텐서플로우 임포트  
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784])
# 784차원의 벡터로 단조화된 이미지 를 텐서로 표현하기. 
W = tf.Variable(tf.zeros([784, 10]))
#784차원의 이미지벡터를 곱하여 10차원 벡터의 증거를 만들기. 
b = tf.Variable(tf.zeros([10]))
#출력에 더하기. 
y = tf.nn.softmax(tf.matmul(x, W) + b)
"""
tf.matmul(x, W)로  x와  W를 곱해서, x가 여러 입력으로 구성된 2D텐서일 경우를 다룰 수 있게 하고 b를 넣고 , tf.nn.softmax 적용하기. 이게 우리 모델을 구현한 것이다. 
"""
y_ = tf.placeholder(tf.float32, [None, 10])
#교차엔트로피를 구현하려는 목적으로  정답을 입력하기 위한 새 placeholder 추가하기.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#tf.log(y)는  y각원소 로그값의 계산 값이다.  그리고 y_의 각 원소들에 각가에 해당한 다.  이 값들을 곱하고 tf.reduce_sum은 텐서의 모든 원소를 더하는 것이다. 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#경사하강법 -- 텐서플로우가 각각의 변수들의 비용을 줄이는 방향으로 수정하느 간단한 방벚 --알고리즘을 이용하여 교차 엔트로피를 최소화하도록 명령하기. 
# Session
init = tf.initialize_all_variables()
#변수들을 초기화하는 작업을 하기. 

sess = tf.Session()
#세선을 선언하기
sess.run(init)
#세션의 모델을 시작하고 ㅡ 변수들을 초기화하는 작업을 실행하기 

# Learning
for i in range(1000):
  #1000번 반복하기 설정하기
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #학습세트로부터 100개의 무작이 데이터들의 일괄처리들을 가져오기 . 
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#placeholders 를 대쳏기 위한 일괄처리 데이터에 train_step  피딩을 실행하기 
# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#특정한 축을 따라 가장 큰  원소의 색인을 알려주는 함수값이 실제와 맞았는지 확인하기 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#부정소숫점을 캐스팅한 후 평균값을 구하여 얼마나 많은 비율로 맞았는지 확인하기.
# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#정확도를 확인하기.