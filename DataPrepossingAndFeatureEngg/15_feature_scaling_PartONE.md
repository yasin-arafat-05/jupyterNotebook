---

# Feature scaling:

---

[video_Link](https://youtu.be/85je4aCdYcE?feature=shared)


<br>

## What is feature?

`In machine learning, a feature (also known as an attribute or predictor) refers to an individual measurable property or characteristic of the data that is used as input for the machine learning model. Features are essentially the variables or columns in the dataset that contain relevant information about the observations or instances being studied.`

![Alt text](image.png)

ছবিতে male height in feet, female heigt in feet and life span in year এই ৩ টা হচ্ছে ৩ টা feature । 

**Scale:** Scale হচ্ছে (lowest value ~ column এর height value) । 

- In **male height in feet** column scale হচ্ছে (0 ~ 8.5) । 
- In **female height in feet** column এর scale হচ্ছে (0 ~ 170) । 
- In **Life span in year** column এর scale হচ্ছে (0 ~ 41) । 

# What is feature scaling?

![Alt text](image-1.png)

- **Feature scaling** হচ্ছে একটা method যার মাধ্যমে আমরা numerical feature কে scale করে (0-1) বা (-1~1) এ নিয়ে যায় । আগের উপরের ছবিতে numerical feature গুলোর unit আলাদা আলাদা ছিল(feet,year) । 

- এইটা data preprocessing এর last step । 

- আমরা feature scaling independent variable এর উপর apply করি । 

- আর আমরা feature scaling train data এর উপর apply করি । আর train and test data এর উপর আমরা যেই feature scaling করলাম সেইটা apply করবো বা feature transform করবো। 

# Why feature scaling?

- আমাদের machine learning মডেল কোন feature এর unit বুঝতে পারে না । তারা শুধু numerical value বুঝতে পারে তাই আমাদের সব feature গুলোর value একই range এর মধ্যে থাকলে এরা ভালো output দিতে পারবে । 

![Alt text](image-2.png)

- আমাদের সকল machine learning algorithm কে scaling করা ডাটা দিতে হয় না । কিছু কিছু algorithm আছে তাদের কে শুধুই feature scaling করা ডাটা দিতে হয় । 

![Alt text](image-3.png)

`ছবিতে যেইসব ml algorithrm আছে তাদের সবাই কে feature scaling করা ডাটা হয় । `

# Type of feature scaling:

![Alt text](image-4.png)

**আমরা maximum Standard Scaler and Min Max Scaler এই দুইটা feature scaling  সবচেয়ে বেশি ব্যবহার করি ।**

