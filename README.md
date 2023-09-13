# SHERE-2_from_appro_to_mass

*Первичные параметры*

1. Тип первичной частицы;
2. Энергия первичной частицы;
3. Модель ядро-ядерного взаимодействия;
4. Модель атмосферы;
5. Высота расположения детектора;
6. Угол падения оси ливня / зенитный угол.
   
Известны х,у -- координаты центров ФЭУ в телескопе.

В результате моделирования получаются числа фотоэлектронов в каждом из 109 ФЭУ (I).

Комбинация x,y,I представляет из себя образ или функцию пространственного распределения (ФПР, оно же LDF).

# I Аппроксимация ФПР

В аппроксимации учтен фон и три вида оптических искажений, подробности о которых можно прочесть в статье Клеманс http://uzmu.phys.msu.ru/abstract/2022/4/2241602/

Функция аппроксимации:

$$\frac{p_0^2}{(1+p_1  r+p_2  r^2+p_3  r^{1.5})^2  (1+p_4  r^s )}$$

## Входные данные

В качестве входных файлов выступают "mosaic_hits*". 

Значения первичных параметров меняются в файле "Path_to_data". 

## Запуск

Для запуска необходимо запустить файл "Launch_appro" (Аппроксимация выполняется в файле "Appro_mh_with_BG_square_with_aberr").

## Выходные реузльтаты

Выходными файлами являются файлы вида "param_bg_m01_q2_10PeV_10-20_900m_Fe.txt", в которых записаны параметры аппркосимирующей функции p0,p1,p2,p3,p4,s и некоторые другие полезные значения. А именно в оглавлении файла присутствует строка ",idx,fcn,p0,p1,p2,p3,p4,s,x0,y0,I_max,sum,Rc_snow,Int,err_Int". Где idx $-$ индекс события mosaic_hits* (если клон №12 события №5, то индекс будет равен 512); fcn $-$ среднеквадратичное отклонение аппроксимации от смоделированных данных; далее идут параметры аппроксимирующей функции; x0,y0 $-$ координаты центра оси ливня на мозаике ФЭУ, рассчитанные в процессе аппроксимации; I_max $-$ максимальное число фотоэлектронов в ФЭУ; sum $-$ суммарное число фотоэлектронов во всех ФЭУ; Rc_snow -- расстояние от оси детектора до центра оси ливня на снегу; Int $-$ интеграл по фПР от 0 до 200мм; err_Int $-$ ошибка этого интеграла.

Файлы "param_bg_*" появляются в репозитории "shpere-2_mass_method\Sphere-2_result\appro_param_with_bg\m01\q2\10PeV\10-20\900m", где m01 $-$ это модель атмофсеры, q2 $-$ модель ядро-ядерного взаимодействия, 10PeV $-$ энергия первичной частицы, 10-20 $-$ зенитные углы, 900m $-$ высота расположения детектора над снежной поверхностью.

# II Оптимизация критерия (поиск радиусов критерия)

В случае, если радиусы, необходимые для расчета критерия неизвестны, необходимо произвести оптимизацию. 


<table>
    <thead>
        <tr>
            <th>Energy</th>
            <th>Angle</th>
            <th>h</th>
            <th>q1</th>
            <th>pN</th>
            <th>NFe</th>
            <th>r1</th>
            <th>r2</th>
            <th>r3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=8 align="center">10 PeV</td>
            <td rowspan=4 align="center">10-20</td>
            <td rowspan=2 align="center">500</td>
            <td align="center">q1</td>
            <th align="center">0.4 </th>
            <th align="center">0.3 </th>
           <td rowspan=2 align="center">100</td>
           <td rowspan=2 align="center">120</td>
           <td rowspan=2 align="center">200</td>
        </tr>
        <tr>
            <td align="center">q2</td>
            <th align="center">0.33 </th>
            <th align="center">0.32 </th>
        </tr>       
        <tr>
            <td rowspan=2 align="center">900</td>
            <td align="center">q1</td>
            <th align="center">0.36 </th>
            <th align="center">0.26 </th>
           <td rowspan=2 align="center">80</td>
           <td rowspan=2 align="center">110</td>
           <td rowspan=2 align="center">170</td>
        </tr>       
         <tr>
            <td align="center">q2</td>
            <th align="center">0.31 </th>
            <th align="center">0.31 </th>
        </tr>       
        <tr>
            <td rowspan=4 align="center">15</td>
            <td rowspan=2 align="center">500</td>
            <td align="center">q1</td>
           <th align="center">R3 </th>
            <th align="center">R3 </th>
           <td rowspan=2 align="center">100</td>
           <td rowspan=2 align="center">120</td>
           <td rowspan=2 align="center">200</td>
        </tr>
         <tr>
            <td align="center">q2</td>
            <th align="center">R3 </th>
            <th align="center">R3 </th>
        </tr>       
        <tr>
            <td rowspan=2 align="center">900</td>
            <td align="center">q1</td>
           <th align="center">R3 </th>
            <th align="center">R3 </th>
           <td rowspan=2 align="center">100</td>
           <td rowspan=2 align="center">120</td>
           <td rowspan=2 align="center">200</td>
        </tr>       
         <tr>
            <td align="center">q2</td>
            <th align="center">R3 </th>
            <th align="center">R3 </th>
        </tr>   
    </tbody>
</table>

**Если радиусы известны, то этот пункт можно пропустить.**

Если нет времени на оптимизацию, то для одномерного критерия в среднем хороши радиусы: $r1=,r2=$, а для двумерного критерия: $r1=,r2=,r3=$



# III Оценка средней массы

## Входные данные

На вход подаются параметры аппроксимации, [выходные данные аппроксимации](# I Аппроксимация ФПР)

## Запуск
Для запуска необходимо запустить файл "Launch_aver_mass".

## Выходные результаты

Оценка средней массы производится с помощью файла "optimization_area_q1&q2-Copy3-top-with_BG_h=900".

Для запуска необходимо запустить файл "Launch_aver_mass".

На вход подаются параметры аппроксимации, 































