#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from matplotlib.image import imread


# In[2]:


data_dir='E:\ML_DL_DATASETS\cell_images'


# In[3]:


os.listdir(data_dir)


# In[4]:


train_path=data_dir+'\\train\\'
test_path=data_dir+'\\test\\'


# In[5]:


os.listdir(train_path)


# In[6]:


os.listdir(train_path+'\\parasitized\\')[0]


# In[7]:


img=imread(train_path+'\\parasitized\\'+'C100P61ThinF_IMG_20150918_144104_cell_162.png')
plt.grid(False)
plt.imshow(img)


# In[8]:


img.shape


# In[9]:


os.listdir(train_path+'\\uninfected\\')[0]


# In[10]:


healthy_cell=imread(train_path+'\\uninfected\\'+os.listdir(train_path+'\\uninfected\\')[0])
plt.grid(False)
plt.imshow(healthy_cell)


# In[11]:


dim1=[]
dim2=[]
color=[]
#for file in os.listdir(test_path+'uninfected\\'):
   # imgg=imread(test_path+'uninfected\\'+file)
   # d1,d2,color=imgg.shape
   # dim1.append(d1)
   # dim2.append(d2)
#for image_filename in os.listdir(test_path+'\\uninfected'):
    
    #img = imread(test_path+'\\uninfected'+'\\'+image_filename)
    #d1,d2,colors = img.shape
    #dim1.append(d1)
    #dim2.append(d2)    
for image_filename in os.listdir(test_path+'\\uninfected'):
    
    imgg = imread(test_path+'uninfected'+'\\'+image_filename)
    d1,d2,colors = imgg.shape
    dim1.append(d1)
    dim2.append(d2)    


# In[12]:


sns.jointplot(dim1,dim2,cmap='accent')


# In[13]:


print(np.mean(dim1),np.mean(dim2))


# In[14]:


image_shape=(130,130,3)


# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[16]:


img_gen=ImageDataGenerator(rotation_range=20,
                           width_shift_range=0.1,
                           height_shift_range=0.1,rescale=1/255,
                           shear_range=0.1,
                           fill_mode='nearest',
                           horizontal_flip=True,
                           zoom_range=0.1)


# In[17]:


plt.imshow(img_gen.random_transform(img))


# In[18]:


plt.imshow(img)


# In[19]:


img_gen.flow_from_directory(train_path)


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Conv2D,Flatten,Dropout,MaxPooling2D


# In[23]:


model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


# In[24]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# model.summary()

# In[25]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',patience=2)`


# In[38]:


img.shape


# In[39]:


train_img_gen=img_gen.flow_from_directory(train_path,
                                          target_size=image_shape[:2],batch_size=16,color_mode='rgb',class_mode='binary')


# In[40]:


test_img_gen=img_gen.flow_from_directory(test_path,batch_size=16,target_size=image_shape[:2],color_mode='rgb',class_mode='binary',shuffle=False)


# In[41]:


train_img_gen.class_indices


# In[42]:


import warnings
warnings.filterwarnings('ignore')


# In[43]:


results = model.fit_generator(train_img_gen,epochs=20,
                              validation_data=test_img_gen,
                             callbacks=[early_stop])


# In[ ]:


losses = pd.DataFrame(model.history.history)


# In[ ]:


losses[['loss','val_loss']].plot()


# In[ ]:


model.metrics_names


# In[ ]:


model.evaluate_generator(test_image_gen)


# In[ ]:


pred_probabilities = model.predict_generator(test_image_gen)


# In[ ]:


pred_probabilities


# In[ ]:


predictions = pred_probabilities > 0.5


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(test_image_gen.classes,predictions))

