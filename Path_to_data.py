chek_appro='off' # or 'on'
glob_add_name='aver_mass_optim_q1&q2'  # Любое имя папки, в которую будут складываться графики
optimization='off' # or 'on'
if optimization=='off':
    best_ia, best_ib, best_ja, best_jb = 0, 0, 0, 0

result_average_mass='off' # or 'on'
N_all=100


model_interaction = 'q2'
model_atmosphere='m01'
Energy_CR = 10       # PeV
Angle_CR = '10-20'    # Degrees
Type_particles_CR = ['p','He','N','S','Fe']
Type_particle_CR = 'p'
h = 500

r1,r2,r3=80,110,170   # Для h=900 





if model_interaction == 'q1':
    model_inter_name ='QGSJET'
if model_interaction == 'q2':
    model_inter_name ='QGSJET_II'

# Директории взятия данных:

# Директория файлов mosaic_hits* :
direct_mosaic_hits = 'C:/Vasilisa/Sphere_new/FPR/mosaic_hits/sphere-2/'+model_inter_name+'/'+str(Energy_CR)+'PeV/'+str(h)+'/'+Type_particle_CR+'/'+Angle_CR+'/'
#direct_mosaic_hits = 'C:/Vasilisa/Sphere_new/FPR/mosaic_hits/sphere-2/QGSJET_II/10PeV/500/p/10-20/'

# Путь к файлам учета сферических аберраций для двух высот:
direct_aberration ='C:/Vasilisa/Sphere_new/aberration_data/'+str(h)+'m/SPHERE-2_'+str(h)+'-detail.txt'

# Путь к файлу с координатами ФЭУ:
direct_coord ='C:/Vasilisa/Sphere_new/FPR/mosaic_pmt_coords.txt'

# Путь к файлу с фоном:
direct_bg = 'C:/Users/2000v/Desktop/shpere-2_mass_method/BG_files'

# Директории вывода результатов программы:

