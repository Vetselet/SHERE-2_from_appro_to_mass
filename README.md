# SHERE-2_from_appro_to_mass

# I Аппроксимация ФПР


Аппроксимация выполняется в файле Appro_mh_with_BG_square_with_aberr.py
параметров аппркосимирующей функции p0,p1,p2,p3,p4,s и некоторых других полезных значений. А именно в оглавлении файла присутствует строка ,idx,fcn,p0,p1,p2,p3,p4,s,x0,y0,I_max,sum,Rc_snow,Int,err_Int. Где idx -- индекс события mosaic_hits* (если клон №12 события №5, то индекс будет равен 512), fcn -- среднеквадратичное отклонение аппроксимации от смоделированных данных, далее идут параметры аппроксимирующей функции, x0,y0 -- координаты центра оси ливня на мозаике ФЭУ, рассчитанные в процессе аппроксимации, I_max -- максимальное число фотоэлектронов в ФЭУ, sum -- суммарное число фотоэлектронов во всех ФЭУ, Rc_snow (R_t в старом варианте) -- расстояние от оси детектора до центра оси ливня на снегу, Int -- интеграл по фПР от 0 до 200мм, err_Int -- ошибка этого интеграла.

