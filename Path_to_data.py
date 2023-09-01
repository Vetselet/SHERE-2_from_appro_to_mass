model_interaction = 'q2'
model_atmosphere='m01'
Energy_CR = 10       # PeV
Angle_CR = '10-20'    # Degrees
Type_particle_CR = 'Fe' 
h = 500

# Директории взятия данных:

# Директория файлов mosaic_hits* :
direct_mosaic_hits = 'C:/Vasilisa/Sphere_new/FPR/mosaic_hits/sphere-2/QGSJET_II/10PeV/500/Fe/10-20/'
# Путь к файлам учета сферических аберраций для двух высот:
direct_aberration ='C:/Vasilisa/Sphere_new/aberration_data/'+str(h)+'m/SPHERE-2_'+str(h)+'-detail.txt'
# Путь к файлу с координатами ФЭУ:
direct_coord ='C:/Vasilisa/Sphere_new/FPR/mosaic_pmt_coords.txt'
# Путь к файлу с фоном:
direct_bg = 'C:/Users/2000v/Desktop/shpere-2_mass_method/BG_files'

# Директории вывода результатов программы:

