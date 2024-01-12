
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib as mpl

# image = plt.imread('OrangeTiger.jpg')
# plt.imshow(image)
# plt.show()

#from PIL import Image

# Открываем изображение
pic = Image.open("76.jpg")
picture = Image.open("76.jpg")
picture1 = Image.open("76.jpg")
#picture = Image.open("02.jpg")
#picture1 = Image.open("02.jpg")
# Получаем rgb составляющую первого пикселя
#r,g,b = picture.getpixel( (0,0) )

# Выводим rgb составляющие
#print("Red: {0}, Green: {1}, Blue: {2}".format(r,g,b))

# Получаем размер изображения (ширину и высоту)
width, height = picture.size
print('Размер изображения: ширина =', width, ', высота =',height)

# задаем количество элементов R на длинну/ширину
#R = int(input("Задайте количество элементов на длинну R = "))
R = 10
prod = R * R
#print('Количество элементов на длинну =', R)

# определяем целое значение шинины/длинны одного элемента
#wi = int(round(width/R, 0))
#he = int(round(height/R, 0))
wi = (width//R)
he = (height//R)

print('Размер элемента: ширина =', wi, ', высота =',he)
#print(wi, he)

# формируем матрицу граничных значенй по X и по Y для элементов
GranXY = np.zeros((2,R+1,) ,int)
# привязывем значения границ к номерам элементов
GranElement = np.zeros((R,R,) ,int)
for x in range(R+1):
    GranXY[0][x] = wi * x
    GranXY[1][x] = he * x
#print(GranXY)
print('Порядок следования элементов на матрице  изображения')
# порядок следования элементов на матрице  изображения
bb = 0
for x in range(R):
    GranXY[0][x] = wi * x
    GranXY[1][x] = he * x
    for y in range(R):
        bb = bb + 1
        GranElement[x][y] = bb
print(GranElement)
print()
# матрица граничных значений  для каждого элемента: [номер элемента, начальный столбец, конечный столбец, начальная строка, конечная строка]
GranElementXY = np.zeros((R*R,4+1,) ,int)
bb1 = -1
for y in range(R):
    for x in range(R):
        bb1 = bb1 + 1
        GranElementXY[bb1][0] = bb1+1
        GranElementXY[bb1][1] = GranXY[0][x]
        GranElementXY[bb1][2] = GranXY[0][x+1]
        GranElementXY[bb1][3] = GranXY[1][y]
        GranElementXY[bb1][4] = GranXY[1][y+1]
#массив элементови и его XY границ
print('Матрица граничных значений  для каждого элемента: [номер элемента, начальный столбец, конечный столбец, начальная строка, конечная строка]')
print(GranElementXY)
print()

# Process every pixel
# Двойной цикл прохода по пикселям с последующим пребразованием
# Расчет суммы значений элементов массива по матрицам R,G,B
k = np.zeros((R*R,) , float)
#print(k)
for i in range(R*R):
    k1 = 0
    xn = GranElementXY[i][1]
    xk = GranElementXY[i][2]
    yn = GranElementXY[i][3]
    yk = GranElementXY[i][4]
    #print('u1', i, xn, xk, yn, yk)
    for x in range(xn, xk):
        for y in range(yn, yk):
            current_color = picture.getpixel((x, y))
            k1 = k1 + current_color[0] + current_color[1] + current_color[2]
            #k1 = k1 + current_color[2]
    k[i] = k1
#print(k)
k = k/max(k)
print('Расчет нормированной суммы значений элементов массива по матрицам R,G,B')
print(k)
print()

# формируем массив для построения матрицы сравнения
m= np.zeros((R*R,R*R,) ,float)
# создаем массив заполненный нулями , он нужен для построения графиков (значения в этом массиве идут через запятую)
KK = [0] * R*R*R*R
MM = [0] * R*R*R*R

# проводим сравннение кждого элемента с каждым, заполняем матрицу ошибок - отличий
kk = -1
for x in range(prod):
    for y in range(prod):
    #for y in range(x+1):
        m[x][y] = k[x] - k[y]
        m[x][y] = round(m[x][y], 5)
        m[x][y] = abs(m[x][y])
        kk = kk + 1
        # массивы для графиков
        KK[kk] = kk
        MM[kk] = m[x][y]
print('Матрица сравнения каждого элемента с каждым')
print(m)
#print(KK)
#print(MM)

# e - задем здесь, чтобы отобразилась на графике
e = 0.07

#перебор массива по убыванию ошибки
MM1 = [0] * R*R*R*R
MM2 = [0] * R*R*R*R
MM3 = [0] * R*R*R*R
MM2 = MM
max1 = np.amax(MM)
MM2 = MM2/max1
for i1 in range(R*R*R*R):
    GG = 0
    for i in range(R * R * R * R):
        if (MM2[i] > GG):
           GG = MM2[i]
    MM1[R * R * R * R-i1-1] = GG
    MM2[R * R * R * R-i1-1] = -1
    MM3[R * R * R * R-i1-1] = e
#print(MM3)
print(MM1)

#Формируем массив введенеим через терминал
#MA=MM
#na = 10
#MA = [int(input()) for i in range(na)]
#print(MA)

#---- Рисуем график
#линейный
from matplotlib import pyplot as plt
#x = KK
#KK = [1, 2, 3]
#print(KK)
#y = m
#MM = [10, 11, 12]
plt.plot(KK, MM1)
plt.plot(KK, MM3)
plt.title("График ошибок")
plt.ylabel('Значение ошибки')
plt.xlabel('Номер элемента')
plt.show()

#гистограмма
#from matplotlib import pyplot as plt
##percentage = [97,54,45,10, 20, 10, 30,97,50,71,40,49,40,74,95,80,65,82,70,65,55,70,75,60,52,44,43,42,45]
##number_of_student = [0,10,20,30,40,50,60,70,80,90,100]
##plt.hist(percentage, number_of_student, histtype='bar', rwidth=0.8)
#plt.hist(MM*10, KK, histtype='bar', rwidth=0.8)
#plt.xlabel('percentage')
#plt.ylabel('Number of people')
#plt.title('Histogram')
#plt.show()


#----


print()
#print('Сравнение элемнтов друг с другом')
#print('--------------- матрица ошибок')
# переводим  строку  в матрицу
AA =np.array(m)
#print(AA)

min = np.amin(m)
max = np.amax(m)
#print(min, max)
m = m/max
#print('---------------матрицу ошибок разделили на ее максимальное значение получили матрицу сравнения')
#print(m)

dataOshibok = np.zeros((R*R,R*R,) ,float)
dataElement = np.zeros((R*R,R*R,) ,int)
#num_rows = len(data)
#num_cols = max(len(row) for row in data)
#если вы используете 2d-списки, то Len(A) возвращает высоту вашей матрицы, тогда Len(A[0]) будет шириной вашей матрицы

data = m
#print(data)
DD = data.shape
rows = DD[0]
cols = DD[1]
#print(rows, cols)

u = dataOshibok
u1 = dataElement
n = m
kmin = 0
for c in range(cols):
    #g = -1
    for ad in range(rows):
        z = 2
        for r in range(rows):
            if (n[r][c] < z) and (n[r][c] < 2):
                gr = r
                gc = c
                z = n[r][c]
        u[ad][c] = z
        u1[ad][c] = gr + 1
        n[gr][gc] = 2

print('---------------перегруппировка матрицы ошибок по возрастанию их значений')
print(u)
#print('---------------соответствующая матрица номеров элементов по возрастанию значений ошибок')
#print(u1)
#print()
#print('Задаем значение уровеня синхронности или меры сходства')

e = float(input("Задайте предельное значение ошибки (< 1, приемр 0.02) e = "))
#e = 0.02
print('e <',e)
for c in range(cols):
    for r in range(rows):
        if (u[r][c] > e) :
            u[r][c] = 2
            u1[r][c] = 0

# значение в матрице ошибок не может быть больше 2
# номер элемента вматрице элементов не может быть  0

#print('---------------выделение элементов соответствующих условию ')
#(u)
print('Выделение элементов соответствующих условию в группы')
print(u1)
#print(GranElementXY)




print()
print('Вывод изображения групп')

print('Задаем номер элемента для которго формируем группу')
Ke = 25
#Ke = R*R
print('Номер элемента =', Ke)
dataugr = np.zeros((1,rows,) ,int)
#print(u1)

for x in range(width):
    for y in range(height):
            current_color1 = picture1.getpixel((x, y))
            picture1.putpixel((x, y), (current_color1[0] * 0, current_color1[1] * 0, current_color1[2] * 0))
            #cur_col = pic.getpixel((x, y))
            #pic.putpixel((x, y), (cur_col[0] * 1, cur_col[1] * 0, cur_col[2] * 0))
#picture1.show()
#picture.show()
#pic.show()
ugr = dataugr
for r in range(R*R):
    if (u1[r][Ke-1] > 0):
        nu = u1[r][Ke-1]
        xn1 = GranElementXY[nu-1][1]
        xk1 = GranElementXY[nu-1][2]
        yn1 = GranElementXY[nu-1][3]
        yk1 = GranElementXY[nu-1][4]
        #print('u1', nu, xn, xk, yn, yk)
        for x in range(xn1,xk1):
            for y in range(yn1,yk1):
                current_color = picture.getpixel((x, y))
                picture1.putpixel((x, y), (current_color[0] , current_color[1] , current_color[2] ))

#print(ko)
r, g, b = picture.getpixel((0, 0))
#print("Red: {0}, Green: {1}, Blue: {2}".format(r, g, b))
picture1.show()


#--- мой расчет конец