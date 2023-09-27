# SHERE-2 from original data approximation to mass

*Первичные параметры*

1. Тип первичной частицы;
2. Высота расположения детектора;
3. Угол падения оси ливня / зенитный угол.
4. Энергия первичной частицы;
7. Модель ядро-ядерного взаимодействия;
8. Модель атмосферы;
   
Известны х,у -- координаты центров ФЭУ в телескопе.

В результате моделирования получаются числа фотоэлектронов в каждом из 109 ФЭУ (I).

Комбинация x,y,I представляет из себя образ / функцию пространственного распределения (ФПР, оно же LDF).

# I Аппроксимация ФПР

В аппроксимации учтен фон и три вида оптических искажений, подробности о которых можно прочесть в статье Клеманс http://uzmu.phys.msu.ru/abstract/2022/4/2241602/

Функция аппроксимации:

$$\frac{p_0^2}{(1+p_1  r+p_2  r^2+p_3  r^{1.5})^2  (1+p_4  r^s )}$$

## Входные данные

В качестве входных файлов выступают "mosaic_hits*". Файлы из папки "BG_files" для учета фона. Файлы из папки "aberration_data" для учета сферических аберраций.

Значения первичных параметров меняются в файле [Path_to_data](https://github.com/Vetselet/SHERE-2_from_appro_to_mass/blob/main/Path_to_data.py) . 

## Запуск

Для запуска необходимо запустить файл [Launch_appro](https://github.com/Vetselet/SHERE-2_from_appro_to_mass/blob/main/Launch_appro.ipynb) (аппроксимация выполняется в файле [Appro_mh_with_BG_square_with_aberr](https://github.com/Vetselet/SHERE-2_from_appro_to_mass/blob/main/Appro_mh_with_BG_square_with_aberr.py)).

## Выходные результаты

Выходными файлами являются файлы вида "param_bg_m01_q2_10PeV_10-20_900m_Fe.txt", в которых записаны параметры аппркосимирующей функции p0,p1,p2,p3,p4,s и некоторые другие полезные значения. А именно в оглавлении файла присутствует строка ",idx,fcn,p0,p1,p2,p3,p4,s,x0,y0,I_max,sum,Rc_snow,Int,err_Int". Где idx $-$ индекс события mosaic_hits* (если рассматривать клон №12 события №5, то индекс будет равен 512); fcn $-$ среднеквадратичное отклонение аппроксимации от смоделированных данных; далее идут параметры аппроксимирующей функции; x0,y0 $-$ координаты центра оси ливня на мозаике ФЭУ, рассчитанные в процессе аппроксимации; I_max $-$ максимальное число фотоэлектронов в ФЭУ; sum $-$ суммарное число фотоэлектронов во всех ФЭУ; Rc_snow -- расстояние от оси детектора до центра оси ливня на снегу; Int $-$ интеграл по ФПР от 0 до 200мм; err_Int $-$ ошибка этого интеграла.

Файлы "param_bg_*" появляются в директории "shpere-2_mass_method\Sphere-2_result\appro_param_with_bg\m01\q2\10PeV\10-20\900m", где m01 $-$ это модель атмофсеры, q2 $-$ модель ядро-ядерного взаимодействия, 10PeV $-$ энергия первичной частицы, 10-20 $-$ зенитные углы, 900m $-$ высота расположения детектора над снежной поверхностью.

# II Оптимизация критерия (поиск радиусов критерия)

В случае, если радиусы колец ФПР, необходимые для расчета критерия неизвестны, необходимо произвести оптимизацию. 

**Если радиусы известны, то этот пункт можно пропустить.**

В таблице ниже приведены радиусы для некоторых первичных параметров.

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

Если нет времени на оптимизацию, то для одномерного критерия в среднем хороши радиусы: $r1=80, r2=140$, а для двумерного критерия: $r1=80, r2=110, r3=170 $

30 PeV, 15 500 -- 50 80, 900 -- 80 90 

# III Оценка средней массы

## Входные данные

На вход подаются параметры аппроксимации, выходные результаты пункта I Аппроксимация ФПР.

## Запуск

Для запуска необходимо запустить файл [Launch_aver_mass](https://github.com/Vetselet/SHERE-2_from_appro_to_mass/blob/main/Launch_aver_mass.ipynb) (оценка средней массы производится с помощью файла [Average_mass_with_optimization_q1q2](https://github.com/Vetselet/SHERE-2_from_appro_to_mass/blob/main/Average_mass_with_optimization_q1q2.py)).

## Выходные результаты

На выходе генерируются графики пронумерованные в порядке логического следования алгоритма (подробности алгоритма описаны в [дипломе](https://github.com/Vetselet/SHERE-2_from_appro_to_mass/blob/main/%D0%94%D0%B8%D0%BF%D0%BB%D0%BE%D0%BC_%D0%9B%D0%B0%D1%82%D1%8B%D0%BF%D0%BE%D0%B2%D0%B0_2023.pdf)).

Они расположены в директории "shpere-2_mass_method\Sphere-2_result\graphs\m01\10PeV\10-20\900m".
































