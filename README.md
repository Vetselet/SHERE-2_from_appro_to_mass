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

В качестве входных файлов выступают "mosaic_hits*". 

Значения первичных параметров меняются в файле "Path_to_data". 

Для запуска необходимо запустить файл "Launch_appro" (Аппроксимация выполняется в файле "Appro_mh_with_BG_square_with_aberr").

Выходными файлами являются файлы вида "param_bg_m01_q2_10PeV_10-20_900m_Fe.txt", в которых записаны параметры аппркосимирующей функции p0,p1,p2,p3,p4,s и некоторые другие полезные значения. А именно в оглавлении файла присутствует строка ",idx,fcn,p0,p1,p2,p3,p4,s,x0,y0,I_max,sum,Rc_snow,Int,err_Int". Где idx $-$ индекс события mosaic_hits* (если клон №12 события №5, то индекс будет равен 512); fcn $-$ среднеквадратичное отклонение аппроксимации от смоделированных данных; далее идут параметры аппроксимирующей функции; x0,y0 $-$ координаты центра оси ливня на мозаике ФЭУ, рассчитанные в процессе аппроксимации; I_max $-$ максимальное число фотоэлектронов в ФЭУ; sum $-$ суммарное число фотоэлектронов во всех ФЭУ; Rc_snow -- расстояние от оси детектора до центра оси ливня на снегу; Int $-$ интеграл по фПР от 0 до 200мм; err_Int $-$ ошибка этого интеграла.

Файлы "param_bg_*" появляются в репозитории "shpere-2_mass_method\Sphere-2_result\appro_param_with_bg\m01\q2\10PeV\10-20\900m", где m01 $-$ это модель атмофсеры, q2 $-$ модель ядро-ядерного взаимодействия, 10PeV $-$ энергия первичной частицы, 10-20 $-$ зенитные углы, 900m $-$ высота расположения детектора над снежной поверхностью.

# II Оптимизация критерия (поиск радиусов критерия)

В случае, если радиусы, необходимые для расчета критерия неизвестны, необходимо произвести оптимизацию. 

Параметры | pN | NFe | r1 |  r2 |  r3  |
|----------|----------|----------|----------|----------|----------|
| q1, h = 500   | 0.4   | 0.3   | 100    | 120   | 200   |
| q2, h = 500   | 0.33   | 0.32   | 100    | 120   | 200   |
| q1, h = 900   | 0.36   | 0.26   | 80    | 110   | 170   |
| q2, h = 900   | 0.31   | 0.31   | 80    | 110   | 130   |


<table>
    <thead>
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
            <th>Column 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=8 align="center">10 PeV</td>
            <td rowspan=4 align="center">10-20</td>
            <td rowspan=2 align="center">500</td>
            <td align="center">q1</td>
        </tr>
         <tr>
            <td rowspan=4 align="center">15</td>
            <td rowspan=2 align="center">500</td>
            <td align="center">q1</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">q2</td>
            <td align="center">q22</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">15</td>
            <td rowspan=2 align="center">500</td>
            <td align="center">q1</td>
        </tr>
        <tr>
            <td align="center">q2</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
            <th>Column 3</th>
            <th>Column 4</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=8 align="center">R0 Text</td>
            <td rowspan=4 align="center">R1 Text</td>
            <td rowspan=2 align="center">R2 Text A</td>
            <td align="center">R3 Text A</td>
        </tr>
        <tr>
            <td align="center">R3 Text B</td>
        </tr>       
        <tr>
            <td rowspan=2 align="center">R2 Text B</td>
            <td align="center">R3 Text C</td>
        </tr>       
         <tr>
            <td align="center">R3 Text D</td>
        </tr>       
        <tr>
            <td rowspan=4 align="center">R1 Text B</td>
            <td rowspan=2 align="center">R2 Text B</td>
            <td align="center">R3 Text A</td>
        </tr>
         <tr>
            <td align="center">R3 Text B</td>
        </tr>       
        <tr>
            <td rowspan=2 align="center">R2 Text B</td>
            <td align="center">R3 Text C</td>
        </tr>       
         <tr>
            <td align="center">R3 Text D</td>
        </tr>    

    </tbody>
</table>

**Если радиусы известны, то этот пункт можно пропустить.**

Если нет времени на оптимизацию, то для одномерного критерия в среднем хороши радиусы: $r1=,r2=$, а для двумерного критерия: $r1=,r2=,r3=$



# III Оценка средней массы











