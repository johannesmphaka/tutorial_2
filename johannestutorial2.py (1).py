
# coding: utf-8

# In[41]:


#problem 1

from numpy.fft import fft,ifft

import numpy as np

from matplotlib import pyplot as plt

def myshift(x,n=0):

    vec=0*x

    vec[n]=1

    vecft=fft(vec)

    xft=fft(x)

    return np.real(ifft(xft*vecft))

if __name__=='__main__':

    x=np.arange(-10,10,0.1)

    sigma=2

    y=np.exp(-0.5*x**2/sigma**2)

    yshift=myshift(y,len(y)//2)

    

    plt.ion()

    plt.plot(x,y)

    plt.plot(x,yshift)


# In[11]:



#provlem 2 

from numpy.fft import fft,ifft

import numpy as np

from matplotlib import pyplot as plt

def mycorrelation(x,y):

    assert(x.size==y.size)  

    xft=fft(x)

    yft=fft(y)

    yftconj=np.conj(yft)

    return np.real(ifft(xft*yftconj))

if __name__=='__main__':

        x=np.arange(-20,20,0.1)

        sigma=2

        y=np.exp(-0.5*x**2/sigma**2)

        y_correlation=mycorrelation(y,y)

        plt.plot(x,y_correlation)

        plt.show()


# In[34]:


#problem 3

def myshift(x,n=0):
    
    vec=0*x  #make a vector of zeros 
    vec[n]=1 # replace the first point with 1

    vecft=fft(vec)

    xft=fft(x)

    return np.real(ifft(xft*vecft))

def mycorrelation(x,y):

    assert(len(x)==len(y))  #if the vectors are different sizes, get grumpy

    x_ft=fft(x)

    y_ft=fft(y)

    yft_conj=np.conj(y_ft)

    return np.real(ifft(x_ft*yft_conj))

if __name__=='__main__':

        x=np.arange(-20,20,0.1)

        sigma=2

        y=np.exp(-0.5*x**2/sigma**2)

        ycorrelation=mycorrelation(y,y)

        yshift=myshift(y,len(y)//4)

        yshiftcorrelation=mycorrelation(yshift,yshift)

        meanerr=np.mean(np.abs(ycorrelation-yshiftcorrelation))

        print ('mean difference between the two correlation functions is ' + repr(meanerr))

        plt.plot(x,ycorrelation)

        plt.plot(x,yshiftcorrelation)        

        plt.show()


# In[35]:


# problem 4  

def conv_nowrap(x,y):

    assert(len(x)==len(y))

    xx=np.zeros(2*len(x))

    xx[0:len(x)]=x

    yy=np.zeros(2*len(x))

    yy[0:len(x)]=y

    xxft=fft(xx)

    yyft=fft(yy)

    vector=np.real(ifft(xxft*yyft))

    return vector[0:len(x)]
if __name__=='__main__':

    x=np.arange(-20,20,0.1)

    sigma=2

    y=np.exp(-0.5*x**2/sigma**2)

    y=y/y.sum()

    yconv=conv_nowrap(y,y)

    plt.plot(x,y)

    plt.plot(x,yconv)

    plt.show()



# In[39]:


#problem 5  
class Complex:
    def __init__(self,r=0.0,i=0.0):
        self.r=r
        self.i=i
    def absl(self):
        t=np.sqrt(self.r**2+self.i**2)
        return y
    def sub(self,x):
        rp = self.r - x.r
        ip = self.i - x.i
        return Complex(rp,ip)
    def add(self,y):
        rp2 = self.r + y.r
        ip2= self.i + y.i
        return Complex(rp2,ip2)
    def mult(self,z):
        rp3 = (self.r*z.r)-(self.i*z.i)
        ip3 = (self.r*z.i)+(z.r*self.i)
        return Complex(rp3,ip3)
    def div(self,d):
        rp4 = ((self.r*d.r)+(self.i*d.i))/(d.r**2+d.i**2)
        ip4 = ((self.i*d.r)-(self.r*d.i))/(d.r**2+d.i**2)
        return Complex(rp4,ip4)
a=Complex(2,3) 
b=Complex(2,4)
z1=b.sub(a) 
z2 =b.add(a)
z3 = a.mult(b)
z4 = a.div(b)
print(a.absl())
print((z1.r,z.i), 'results for addition')
print((z2.r,z2.i),'results for subtraction')
print((z3.r,z3.i),'results for multiplication')
print((z4.r,z4.i),'results for division')
               

