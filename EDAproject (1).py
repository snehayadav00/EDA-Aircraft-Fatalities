#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.chdir("D:/CSV files")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.read_csv('Aircraft_Incident_Dataset.csv')
data1.head()


# In[8]:


data1.shape
# This shows the number of rows and number of features in our data


# In[15]:


plt.rcParams['figure.figsize']=[15,8]
sns.heatmap(data1.isnull())
plt.title('Missing Value Plot',fontsize=30)
plt.xlabel("Features",fontsize=18)
plt.ylabel('Index Number',fontsize=18)
plt.show()


# In[9]:


data2 = data1.copy()    # Making a copy of data1 so that any changes done are not reflected back in my original data

# trying to simplify the Incident_Date Column
Incident_date,Incident_month,Incident_year=[],[],[]
for i in data2['Incident_Date']:
    j=i.split('-')
    Incident_date.append(j[0])
    Incident_month.append(j[1].lower())
    Incident_year.append(j[2])
data2['Incident_date'] = Incident_date
data2['Incident_month'] = Incident_month
data2['Incident_year'] = Incident_year
data2.head()


# In[10]:


# trying to simplify the Date column

Date_day=[]
for i in data2['Date']:
    j = i.split(' ')
    Date_day.append(j[0])
print(np.unique(Date_day))
data2['Day'] = Date_day
data2.head()


# In[11]:


print(np.unique(data2['Incident_month']))
data2['Incident_month'].value_counts()
data2[data2['Incident_month']=='14']
data2['Incident_month'].replace('14','feb',inplace=True)
data2[data2['Incident_month']=='20']
data2['Incident_month'].replace('20','aug',inplace=True)
data2['Incident_month'].value_counts()

data2=data2.drop(data2[data2['Incident_month']=='???'].index,axis=0)


# In[12]:


data2['Incident_date'].value_counts()
data2=data2.drop(data2[data2['Incident_date']=='??'].index,axis=0)

year=np.unique(data2['Incident_year'])
year

data2['Day'].value_counts()
data2.drop(axis=1,columns=['Incident_Date','Date'],inplace=True)
data2.head()


# In[16]:


data2['Time'].value_counts()
data2[data2['Time'].isnull() == True].shape
# As we cannot determine what ca/ca./PST etc is in reality and the amount of null is very high 
# so we treat it by removing this particular column


# In[17]:


Arit_date,Arit_month,Arit_year=[],[],[]
for i in data2['Arit']:
    j=i.split('-')
    Arit_date.append(j[0])
    Arit_month.append(j[1].lower())
    Arit_year.append(j[2])
np.unique(Arit_month)
if data2['Incident_date'].tolist() == Arit_date:
    print("Same")
# As the Arit dates is also same as Incident dates, therefore dropping Arit column too
data2.drop(columns=['Time','Arit'],axis=1,inplace=True)
data2.head()


# In[21]:


def fat_occup(data):
    l1,l2=[],[]
    for i in data:
        j=i.split('/')
        k=j[0].split(':')
        l1.append(k[1])
        f = j[1].split(':')
        l2.append(f[1])
    return l1,l2
data2['Crew_fatal'],data2['Crew_Occup'] = fat_occup(data2['Onboard_Crew'])
data2['Pass_fatal'],data2['Pass_Occup'] = fat_occup(data2['Onboard_Passengers'])

data2['Crew_fatal'].replace('',0,inplace=True)
data2['Crew_Occup'].replace('',0,inplace=True)
data2['Pass_fatal'].replace('','0',inplace=True)
data2['Pass_Occup'].replace('','0',inplace=True)

data2.head()


# In[22]:


data2['Ground_Casualties'].fillna('0',inplace=True)
data2['Collision_Casualties'].fillna('0',inplace=True)
data2['Ground_Casualties'].head()
data2['Ground_Casualties'].value_counts()
data2.info()


# In[23]:


#data2['Ground_Casualties']
l1,l2=[],[]
allowed_chars = set('1234567890')
for i in data2['Ground_Casualties']:
    if set(i).issubset(allowed_chars)==False:
        j=i.split(' ')
        l1.append(j[1])
    else:
        l1.append(i)
data2['Ground_Casualties']=l1
for i in data2['Collision_Casualties']:
    if set(i).issubset(allowed_chars)==False:
        j=i.split(' ')
        l2.append(j[1])
    else:
        l2.append(i)
data2['Collision_Casualties']=l2
data2.head()


# In[24]:


plt.rcParams['figure.figsize'] = [20,10]


# In[25]:


data2['Aircaft_Nature'].value_counts()


# In[26]:


df = data2.groupby('Aircaft_Nature')
l1=[]
l2=[]
l3=[]
for name,grp in df:
    #print(name)
    l1.append(name)
    l2.append(sum(grp['Fatalities']))
    l3.append(len(grp))


# In[27]:


plt.rcParams['figure.figsize'] = [13,10]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.xticks(rotation=90,fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.xlabel('Aircraft Nature',fontsize=20,color='red')
plt.ylabel('Number of Fatalities',fontsize=20,color='red')
plt.title('Fatalities against each Aircraft Nature',fontsize=30,y=1.1)
plt.show()


# In[68]:


plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.xticks(rotation=90,fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.xlabel('Aircraft Nature',fontsize=20,color='red')
plt.ylabel('Number of Incidents',fontsize=20,color='red')
plt.title('Incidents against each Aircraft Nature',fontsize=30,y=1.1)
plt.show()


# In[69]:


l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
plt.rcParams['figure.figsize'] = [15,10]
plt.plot(l1,l4,color='darkmagenta',marker='o',markerfacecolor='black')
plt.title('Average fatalities V/s Aircraft Nature',fontsize=25,color='black',y=1.02)
plt.xticks(rotation=90,fontsize=13,color='black')
plt.yticks(fontsize=13,color='black')
plt.xlabel('Aircraft Nature',fontsize=20,color='darkblue')
plt.ylabel('Average Fatalities per incident',fontsize=20,color='darkblue')
plt.show()


# In[ ]:





# In[16]:


#data2['Aircaft_Nature'].replace('-','Unknown',inplace=True)
#sns.set(rc={'figure.figsize':(10,10)})
#for i in np.unique(data2['Aircaft_Nature']):
#    data3=data2[data2['Aircaft_Nature']==i]
#    #print(data3['Fatalities'].value_counts())
#    sns.countplot(x='Fatalities',data=data3)
#    plt.title(i)
#    plt.xticks(rotation=90)
#    plt.show()


# Plotting number of fatalities according to every Aircraft Nature

# In[17]:


#a = np.unique(data2['Aircaft_Nature'])
#b={}
#for i in a:
#    data3 = data2[data2['Aircaft_Nature']==i]
#    c=sum(data3['Fatalities'])
#    print(i,c)
#    b[i]=c

#fig=plt.figure()
#fig.set_size_inches(12,10)
#plt.bar(*zip(*b.items()))
#label=b.items()
#for i in range(0,len(label)):
#    plt.text(x=i,y=label[i],s=label[i],size=9,rotation=90)
#plt.xticks(rotation=90)
#plt.show()


# In[18]:


data2['Aircaft_Operator'].value_counts()
a=np.unique(data2['Aircaft_Operator'])
b=0
l1=[]
l2=[]
l3=[]
for i in a:
    data3=data2[data2['Aircaft_Operator']==i]
    c=sum(data3['Fatalities'])
#    print(i,c)
    if c>500:
        l1.append(i)
        l2.append(c)
        #print(i,c)
        d=data2[data2['Aircaft_Operator']==i].shape[0]
        l3.append(d)
        b=b+1
#print(b)


# In[19]:


plt.rcParams['figure.figsize'] = [15,10]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.xticks(rotation=90,fontsize=12,color='blue')
plt.yticks(fontsize=12,color='blue')
plt.xlabel('Aircraft Operator',fontsize=20,color='red')
plt.ylabel('Number of Fatalities',fontsize=20,color='red')
plt.title('Fatalities against each Aircraft Operator',fontsize=30)
plt.show()


# In[20]:


plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=12, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.xticks(rotation=90,fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.xlabel('Aircraft Operator',fontsize=20,color='red')
plt.ylabel('Number of Incidents',fontsize=20,color='red')
plt.title('Incident against each Aircraft Operator',fontsize=30)
plt.show()


# In[21]:


# Aircraft_Operator:
l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
l4
plt.plot(l1,l4,color='blue',marker='o',markerfacecolor='red')
plt.title('Average fatalities v/s Aircraft_Operator',fontsize=25,color='black',y=1.02)
plt.xticks(rotation=90,fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')
plt.xlabel('Aircraft Operator',fontsize=18,color='darkblue')
plt.ylabel('Average Fatalities per incident',fontsize=18,color='darkblue')
plt.show()


# In[22]:


data2.head()


# In[23]:


data2['Aircaft_Damage_Type'].value_counts()


# In[24]:


# next rows contain in detail
#plt.rcParams['figure.figsize'] = [15,8]
#plots=sns.countplot(x=data2['Aircaft_Damage_Type'])
#for bar in plots.patches:
#    plots.annotate(format(bar.get_height(), '.2f'),
#                       (bar.get_x() + bar.get_width() / 2,
#                        bar.get_height()), ha='center', va='bottom',
#                       size=15, xytext=(0, 8),
#                       textcoords='offset points',rotation=90)
#plt.xticks(rotation=90,fontsize=15)
#plt.title('')
#plt.show()


# In[46]:


group=data2.groupby('Aircaft_Damage_Type')
l1=[]
l2=[]
l3=[]
for name,grp in group:
    print(name)
    l1.append(name)
    l2.append(sum(grp['Fatalities']))
    l3.append(len(grp))
    #sns.countplot(x='Fatalities',data=grp)
    #plt.xticks(rotation=90)
    #plt.show()


# In[47]:


plt.rcParams['figure.figsize'] = [10,8]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
plt.xticks(rotation=90,fontsize=12,color='blue')
plt.yticks(fontsize=12,color='blue')
plt.xlabel('Aircraft Damage Type',fontsize=18,color='red')
plt.ylabel('Number of Fatalities',fontsize=18,color='red')
plt.title('Fatalities v/s Aircraft Damage Type',fontsize=30,loc='right',y=1.02)
plt.show()


# In[48]:


plt.rcParams['figure.figsize'] = [10,8]
plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
plt.xticks(rotation=90,fontsize=12,color='blue')
plt.yticks(fontsize=12,color='blue')
plt.xlabel('Aircraft Damage Type',fontsize=18,color='red')
plt.ylabel('Number of Incidents',fontsize=18,color='red')
plt.title('Incidents v/s Aircraft Damage Type',fontsize=30,y=1.02)
plt.show()


# In[49]:


# Aircraft_Damage_Type:
l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
l4
plt.plot(l1,l4,color='darkmagenta',marker='o',markerfacecolor='black')
plt.xticks(rotation=90,fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')
plt.xlabel('Aircraft Damage Type',fontsize=20,color='darkblue')
plt.ylabel('Number of Fatalities per Incident',fontsize=20,color='darkblue')
plt.title('Average fatalities v/s Aircraft Damage Type',fontsize=30,y=1.02)
plt.show()


# In[29]:


group=data2.groupby('Incident_month')
l1=[]
l2=[]
l3=[]
for name,grp in group:
    print(name)
    l1.append(name)
    l2.append(sum(grp['Fatalities']))
    #sns.countplot(x='Fatalities',data=grp)
    #plt.xticks(rotation=90)
    #plt.show()
    l3.append(len(grp))


# In[30]:


plt.rcParams['figure.figsize'] = [12,8]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.xticks(fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.xlabel('Months',fontsize=20,color='red')
plt.ylabel('Number of Fatalities',fontsize=20,color='red')
plt.title('Fatalities in each Month',fontsize=30,y=1.04)
plt.show()


# In[31]:


plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.title('Incidents in each month',fontsize=30,x=0.5,y=1.1)
plt.xlabel('Months',fontsize=20,color='red')
plt.ylabel('Number of Incidents',fontsize=20,color='red')
plt.xticks(fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.show()


# In[32]:


# Incident_month:
l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
l4
plt.plot(l1,l4,marker='o',color='blue',markerfacecolor='red')
plt.title('Average Fatalities V/s Months',fontsize=30,y=1.02)
plt.xlabel('Months',fontsize=20,color='darkblue')
plt.ylabel('Average Fatalities in each Incident',fontsize=20,color='darkblue')
plt.xticks(fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')
plt.show()


# In[ ]:





# In[50]:


group = data2.groupby('Day')
l1=[]
l2=[]
l3=[]
for key, dat in group:
    l1.append(key)
    l2.append(sum(dat['Fatalities']))
    l3.append(len(dat))


# In[53]:


plt.rcParams['figure.figsize'] = [10,10]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.title('Fatalities v/s Day',fontsize=30,y=1.05)
plt.xlabel('Day of the Week',fontsize=20,color='red')
plt.ylabel('Number of Fatalities',fontsize=20,color='red')
plt.xticks(fontsize=14,color='blue')
plt.yticks(fontsize=14,color='blue')
plt.show()


# In[54]:


plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.title('Incidents V/S Day',fontsize=30,y=1.07)
plt.xlabel('Day of the Week',fontsize=20,color='red')
plt.ylabel('Number of Incidents',fontsize=20,color='red')
plt.xticks(fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.show()


# In[55]:


# Day:
l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
l4
plt.plot(l1,l4,marker='o',color='blue',markerfacecolor='red')
plt.title('Average Fatalities v/s Day',fontsize=30)
plt.xlabel('Day of the Week',fontsize=20,color='darkblue')
plt.ylabel('Average Fatalities per Incident',fontsize=20,color='darkblue')
plt.xticks(fontsize=14,color='black')
plt.yticks(fontsize=14,color='black')
plt.show()


# In[57]:


group = data2.groupby('Incident_date')
l1,l2,l3=[],[],[]
for key, dat in group:
    l1.append(key)
    l2.append(sum(dat['Fatalities']))
    l3.append(len(dat))


# In[58]:


plt.rcParams['figure.figsize'] = [14,10]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.title('Fatalities v/s Dates',fontsize=30,y=1.03)
plt.xlabel('Dates of the Month',fontsize=20,color='red')
plt.ylabel('Number of Fatalities',fontsize=20,color='red')
plt.xticks(fontsize=14,color='blue')
plt.yticks(fontsize=14,color='blue')
plt.show()


# In[59]:


plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)
plt.title('Incidents v/s Dates',fontsize=30,y=1.03)
plt.xlabel('Dates of the Month',fontsize=20,color='red')
plt.ylabel('Number of Incidents',fontsize=20,color='red')
plt.xticks(fontsize=14,color='blue')
plt.yticks(fontsize=14,color='blue')
plt.show()


# In[60]:


# Date:
l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
l4
plt.plot(l1,l4,marker='o',color='blue',markerfacecolor='red')
plt.xlabel('Dates of the Month',fontsize=20,color='darkblue')
plt.ylabel('Average Fatalities per Incident',fontsize=20,color='darkblue')
plt.xticks(fontsize=14,color='black')
plt.yticks(fontsize=14,color='black')
plt.title('Average fatalities on each date')
plt.show()


# In[61]:


data2['Aircraft_Phase'].replace('()','Unknown (UNK)',inplace=True)
data2['Aircraft_Phase'].replace('(CMB)','Climbing (CMB)',inplace=True)
group = data2.groupby('Aircraft_Phase')
l1=[]
l2=[]
l3=[]
for key, dat in group:
    l1.append(key)
    l2.append(sum(dat['Fatalities']))
    l3.append(len(dat))
#print(l1,l2,l3)


# In[62]:


plt.rcParams['figure.figsize'] = [10,10]
plots=sns.barplot(x=l1,y=l2)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)

plt.title('Fatalities v/s Aircraft Phases',fontsize=30,y=1.12)
plt.xlabel('Aircraft Phase',fontsize=20,color='red')
plt.ylabel('Number of Fatalities',fontsize=20,color='red')
plt.xticks(rotation=90,fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.show()


# In[63]:


plots=sns.barplot(x=l1,y=l3)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='bottom',
                       size=15, xytext=(0, 8),
                       textcoords='offset points',rotation=90)

plt.title('Incidents v/s Aircraft Phase',fontsize=30,y=1.12)
plt.xlabel('Aircraft Phase',fontsize=20,color='red')
plt.ylabel('Number of Incidents',fontsize=20,color='red')
plt.xticks(rotation=90,fontsize=15,color='blue')
plt.yticks(fontsize=15,color='blue')
plt.show()


# In[64]:


l4=[]
for i in range (0,len(l1)):
    l4.append(l2[i]/l3[i])
l4
plt.plot(l1,l4,marker='o',color='darkmagenta',markerfacecolor='black')
plt.title('Average fatalities v/s Aircraft Phase',fontsize=30)
plt.xlabel('Aircraft Phase',fontsize=20,color='darkblue')
plt.ylabel('Average Fatalities per Incident',fontsize=20,color='darkblue')
plt.xticks(rotation=90,fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')
plt.show()


# In[45]:


#import matplotlib
#for cname, hex in matplotlib.colors.cnames.items():
#    print(cname,hex)

