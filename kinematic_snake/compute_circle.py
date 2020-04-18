# Python3 implementation of the approach  

# This code is contributed by Ryuga 
# https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
# Slightly modified by Noel to work with kinematic snake
  
# Function to find the circle on  
# which the given three points lie  
def findCircle(avg1_com, avg2_com, avg3_com) : 
    from math import sqrt 
    x1, y1 = avg1_com 
    x2, y2 = avg2_com 
    x3, y3 = avg3_com

    x12 = x1 - x2;  
    x13 = x1 - x3;  
  
    y12 = y1 - y2;  
    y13 = y1 - y3;  
  
    y31 = y3 - y1;  
    y21 = y2 - y1;  
  
    x31 = x3 - x1;  
    x21 = x2 - x1;  
  
    # x1^2 - x3^2  
    sx13 = pow(x1, 2) - pow(x3, 2);  
  
    # y1^2 - y3^2  
    sy13 = pow(y1, 2) - pow(y3, 2);  
  
    sx21 = pow(x2, 2) - pow(x1, 2);  
    sy21 = pow(y2, 2) - pow(y1, 2);  

    f = ((sx13) * (x12) + (sy13) * 
          (x12) + (sx21) * (x13) + 
          (sy21) * (x13)) / (2 * ((y31) * (x12) - (y21) * (x13))); 
              
    g = ((sx13) * (y12) + (sy13) * (y12) + 
          (sx21) * (y13) + (sy21) * (y13)) / (2 * ((x31) * (y12) - (x21) * (y13)));  
  
    c = (-pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1);  
  
    # eqn of a circle is (x-h)^2 + (y-k)^2 = r^2
    # expanding out you can get a new eqn for the circle as
    # eqn of circle: x^2 + y^2 + 2*g*x + 2*f*y + c = 0  
    # where the center is (h = -g, k = -f) and  
    # the radius r is r^2 = h^2 + k^2 - c  
    h = -g;  
    k = -f;  
    sqr_of_r = h * h + k * k - c;  
  
    # r is the radius  
    r = sqrt(sqr_of_r)
  
    # print("Centre = (", h, ", ", k, ")");  
    # print("Radius = ", r);  
    return r
  
# Driver code  
if __name__ == "__main__" :  
      
    x1 = 1 ; y1 = 1;  
    x2 = 2 ; y2 = 4;  
    x3 = 5 ; y3 = 3;  
    r = findCircle(x1, y1, x2, y2, x3, y3);  
  
