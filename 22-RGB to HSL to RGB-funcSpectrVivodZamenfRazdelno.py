import numpy as np
from PIL import Image
import immodus


#Ввод изображения
picture = Image.open("76.jpg")
picture.show()
# Определяем размер изображения
width, height = picture.size
print('Размер изображения: ширина =', width, ', высота =',height)


#=====================================================================================================
# Вызовы функций
#M1 = RGBtoHSL(r,g,b) #присвоение локальной переменной возвращаеемого функцией значения
fname1 = "!01-RGB-R.txt" #задал имя файла как переменную
fname2 = "!02-RGB-G.txt" #задал имя файла как переменную
fname3 = "!03-RGB-B.txt" #задал имя файла как переменную
fname4 = "!04-HSL-H.txt" #задал имя файла как переменную
fname5 = "!05-HSL-S.txt" #задал имя файла как переменную
fname6 = "!06-HSL-L.txt" #задал имя файла как переменную
fname7 = "!07-RGB-RO.txt" #задал имя файла как переменную
fname8 = "!08-RGB-GO.txt" #задал имя файла как переменную
fname9 = "!09-RGB-BO.txt" #задал имя файла как переменную
#print(M1)
#ArrayToFile (fname, M1)


# Расчет для каждого элемента из RGB
A1 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A2 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A3 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A4 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A5 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A6 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A7 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A8 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A9 = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A7I = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A8I = np.zeros((height, width), float) # height количество строк, width-количество столбцов
A9I = np.zeros((height, width), float) # height количество строк, width-количество столбцов

zvetVibor = 22 #выбранный цвет H
Delta = 30 #дисперсия
zvetNew = zvetVibor #цвет для замены = исходный

#Нужно доработать Если цвет исходный то и все др значения исходный иначе меняем на нужный цвет
#Хотя иногда нужно изменить только оттенок - тоже нужно доработать
#цвет для замены #заменяется только значение оттенка с исходными праметрами L и S
#для того чтобы полностью изменить цвет нужно менять сразу три параметра H, S, L
#как это сделано для фонового цвета
#zvetNew = 56   #желтый цвет
#NasishNew = 96 #желтый цвет
#JarkFNew = 49  #желтый цвет

Fon = 0 #значение фона для функции замены цвета = освещенность 0-100

zvetFon = 100 #черный
NasishFon = 100 #черный
JarkFon = 100    #черный

#на  изображениях для того чтобы изменить цвет фона на нужный необходимо задавать все три параметра
#zvetFon = 57 #желтый цвет
#NasishFon = 96 #желтый цвет
#JarkFon = 49    #желтый цвет

#zvetFon = 232 #синий цвет
#NasishFon = 96  #синий цвет
#JarkFon = 49 #синий цвет

for y in range(width-0): #y - номер столбца
   for x in range(height-0):  #x - номер строки
       r, g, b = picture.getpixel( (y,x) ) #r,g,b = picture.getpixel( (width-1, height-1) ) - здесь сначала столбцы потом строки
       A1[x][y] = r     #исходные r,g,b
       A2[x][y] = g
       A3[x][y] = b
       M2 = immodus.RGBtoHSL(r, g, b) #преобразование прямое  r,g,b => h,l,s
       H = M2[0][0]
       Sp = M2[0][1]
       Lp = M2[0][2]
       A4[x][y] = H    #получаем массив оттенков равный размеру одного из каналов изображения
       A5[x][y] = Sp   # - насыщенность
       A6[x][y] = Lp   # - яркость
       #преобразование обратное преобразование  h,l,s => rO,gO,bO
       M3 = immodus.HSLtoRGB(H, Sp, Lp)
       A7[x][y] = M3[0][0] #=rO    #получаемые значения в обратном преобразовании
       A8[x][y] = M3[0][1] #=gO    #без изменений - нужно тольк для проверки точности вычисления
       A9[x][y] = M3[0][2] #=bO    #прямого и обратного преобразований на задаваемом изображении (полноценная проверка д.б. на полном спектре)
       # проводим замену цвета
       #Fon = Lp # ранее фон выбран черным Fon = 0, применив Fon = Lp фон будет соответствовать исходному изображению
       HI, SpI, LpI = immodus.HSLFindH(H, Sp, Lp, JarkFon, NasishFon, zvetFon, zvetVibor,zvetNew, Delta)
       #обратное преобразование с одновременной заменой цвета  h,l,s => rOI,gOI,bOI
       M3 = immodus.HSLtoRGB(HI, SpI, LpI)
       A7I[x][y] = M3[0][0] #=rOI
       A8I[x][y] = M3[0][1] #=gOI
       A9I[x][y] = M3[0][2] #=bOI
picture1 = immodus.RGBNew(picture, A7I, A8I, A9I) #заменяем исходные значения в каналах r,g,b => rOI,gOI,bOI на восстановленные
picture1.show()
picture1.save('Result-76.jpg')
print("")

# расчитываем какие цвета есть и сколько их в изображений
SpektrH = np.zeros((1, 361), int)  # нужно задать массив для функции расчета спектра цветов
#SpektrH = SpektrZvFromHSL(H, SpektrH)  # это нужно для анализа спктра цветов изображения

for y in range(width - 0):  # y - номер столбца
    for x in range(height - 0):  # x - номер строки
       kc = int(A4[x][y])
       SpektrH[0][kc] = SpektrH[0][kc] + 1
#return SpektrH

Sf  = [0] * 361 #так сосздается массив со значениями через запятую необходимый для рисования графиков
Sx  = [0] * 361 #так сосздается массив со значениями через запятую необходимый для рисования графиков
for a in range(361): #y - номер столбца
    Sx[a] = a
    Sf[a] = SpektrH[0][a]
#print(Sf)

#---- Рисуем график
#линейный
from matplotlib import pyplot as plt
#x = KK
#KK = [1, 2, 3]
#print(KK)
#y = m
#MM = [10, 11, 12]
plt.plot(Sx, Sf)
#plt.plot(KK, MM3)
plt.title("Цветовой спектр изображения")
plt.ylabel('Количество пикселей каждого цвета')
plt.xlabel('Номер цвета')
plt.show()
#гистограмма
#from matplotlib import pyplot as plt
#percentage = [97,54,45,10, 20, 10, 30,97,50,71,40,49,40,74,95,80,65,82,70,65,55,70,75,60,52,44,43,42,45]
#number_of_student = [0,10,20,30,40,50,60,70,80,90,100]
#plt.hist(percentage, number_of_student, histtype='bar', rwidth=0.8)
#plt.hist(Sx, Sf, histtype='bar', rwidth=0.8)
#plt.xlabel('percentage')
#plt.ylabel('Number of people')
#plt.title('Histogram')
#plt.show()



#####----Проверка на ошибки вычисления. Исходный r сравнивается с восстановленным
print("---Проверка на ошибки вычисления---")
R = A3 - A9
Nma = 0
Nmi = 10000
Sch1 = 0
Sch2 = 0
Sch3 = 0
Sch4 = 0
#максимальная ошибка на всем изображении
#Тест проведен для изображения "0.jpg" это полный спектр - ошибка максимальная составила
# для массива RED 3 ошибка минимальная составила -3
# - это прмерено 3*100/255 = 1,2% это повторяется в 1,6% случаев
# ошибка = 2 повторяется в 8,5% случаев
# ошибка = 1 повторяется в 16,4% случаев
# 1.2 + 8.5 +16.4 =26.1 В 26,1% случаев для максимальной ошибки.
# для минимальной ошибки 17,0% случаев
#Выводы: ошибка не большая но присутствуетв 25% случаев.
#Предположение - часть ошибок связана с окуруглением (это ошибк 1), думаю оставшаяся часть
# связана с формулами....
# максимальная ошибка для массива GREEN равна 4 => 24% элементов с ошибками, минимальная -3=> 11% элементов
# максимальная ошибка для массива BLUE равна 4 => 24% элементов с ошибками, минимальная -3=> 17% элементов
for c in range(width - 0):  # y - номер столбца
   for r in range(height - 0):
       if (R[r][c] > Nma):
           Nma = R[r][c]
       if (R[r][c] < Nmi):
           Nmi = R[r][c]
       if (R[r][c] == -1):
           Sch1 = Sch1+1
       if (R[r][c] == -2):
           Sch2 = Sch2+1
       if (R[r][c] == -3):
           Sch3 = Sch3+1
       if (R[r][c] == -4):
           Sch4 = Sch4+1

print("Максимум ошибки = ", Nma,"Минимум ошибки = ", Nmi,)
print("Количество ошибок равных 1 = ", Sch1," из ",width*height, " в% ", (round(Sch1*100/(width*height)), 0) )
print("Количество ошибок равных 2 = ", Sch2," из ",width*height, " в% ", (round(Sch2*100/(width*height)), 0) )
print("Количество ошибок равных 3 = ", Sch3," из ",width*height, " в% ", (round(Sch3*100/(width*height)), 0) )
print("Количество ошибок равных 4 = ", Sch4," из ",width*height, " в% ", (round(Sch4*100/(width*height)), 0) )

#----Конец проверки появления ошибок при обратном преобразовании для массива (изображения спектра)

#Вызов функций записи массивов в файлы
#временно отменям функцию записи в фал для экономии времени
###ArrayToFile (fname1,fname2,fname3,A1,A2,A3)
###ArrayToFile (fname4,fname5,fname6,A4,A5,A6)
###ArrayToFile (fname7,fname8,fname9,A7,A8,A9)



#### ===== Проверка - сравнение обратного и прямого преобразований  ====
print("----------------------- Проверка")
r = 255
g = 48
b = 189
print("Преобразование прямое RGB => HSL")
print("R=",r,"G=",g,"B=",b)
MMM = immodus.RGBtoHSL(r, g, b)
h2 = int(MMM[0][0])
s2 = int(MMM[0][1])
l2 = int(MMM[0][2])
print("H=",h2,"S=",s2,"L=",l2 )
#print(MMM)
print("-----------------------")
#онлайн калькулятор https://regtool.net/ru/color/rgb_to_hsl/?ysclid=lryk7jft5f689559990

Sp = s2
Lp = l2
H = h2

MMMO = immodus.HSLtoRGB(H,Sp,Lp)
r = int(MMMO[0][0])
g = int(MMMO[0][1])
b = int(MMMO[0][2])
print("Преобразование обратное HSL => RGB ")
print("R=", r, "G=", g, "B=", b)
## - Конец проверок
