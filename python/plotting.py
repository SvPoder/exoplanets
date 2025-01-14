from utils_plots import plot_1Dposterior


# ----------
# Exercise 1
# ----------
#data_dir = "/Users/mariabenito/Desktop/results/ex1/"
#print("ex1 : 1")
#plot_1Dposterior_ex1(0.3, 1., 20, data_dir, plot_f=True)
#plot_1Dposterior_ex1(0.3, 1.8, 20, data_dir, plot_f=True, color="g")
#print("ex1 : 2")
#plot_1Dposterior_ex1(0.3, 1., 20, data_dir, plot_f=False)
#plot_1Dposterior_ex1(0.3, 1.8, 20, data_dir, plot_f=False, color="g")

# ----------
# Exercise 2
# ----------
#data_dir = "/Users/mariabenito/Desktop/results/ex2/"
#print("ex2 : 1")
#plot_1Dposterior_ex2(0.3, 1., 20, data_dir, plot_f=True, plot_g=False)
#plot_1Dposterior_ex2(0.3, 1.8, 20, data_dir, plot_f=True, plot_g=False, color="g")
#print("ex2 : 2")
#plot_1Dposterior_ex2(0.3, 1., 20, data_dir, plot_f=False, plot_g=True)
#plot_1Dposterior_ex2(0.3, 1.8, 20, data_dir, plot_f=False, plot_g=True, color="g")
#print("ex2 : 3")
#plot_1Dposterior_ex2(0.3, 1., 20, data_dir, plot_f=False, plot_g=False)
#plot_1Dposterior_ex2(0.3, 1.8, 20, data_dir, plot_f=False, plot_g=False, color="g")


# ----------
# Exercise 3
# ----------
data_dir = "/hdfs/local/mariacst/exoplanets/results/final_round/all_unc/GC/"
print("1")
plot_1Dposterior(data_dir, 1000, 0.10, 0.10, "ex11", 1., 1.5, 5.)
plot_1Dposterior(data_dir, 1000, 0.10, 0.10, "ex11", 1., 1.5, 20.)
#plot_1Dposterior(data_dir, 100000, 0.10, 0.10, "ex13", 1., 1., 10.)
print("2")
#plot_1Dposterior(data_dir, 1000, 0.10, 0.10, "ex4", 1., 1.5, 5.)
print("3")
data_dir = "/hdfs/local/mariacst/exoplanets/results/final_round/Tmin/"
#plot_1Dposterior(data_dir, 1000, 0.10, 0.10, "ex5", 1., 1., 10.)
print("4")
#plot_1Dposterior(data_dir, 10000, 0.10, 0.10, "ex3", 1., 1., 20.)

# ----------
# Exercise 4
# ----------
#print("ex4")
#plot_1Dposterior_ex(0.3, 1., 20, "/Users/mariabenito/Desktop/results/ex4/", "ex4")
#plot_1Dposterior_ex(0.3, 1.8, 20, "/Users/mariabenito/Desktop/results/ex4/", "ex4", color="g")


# ----------
# Exercise 5
# ----------
#print("ex5")
#plot_1Dposterior_ex(0.3, 1., 20, "/Users/mariabenito/Desktop/results/ex5/", "ex5")
#plot_1Dposterior_ex(0.3, 1.8, 20, "/Users/mariabenito/Desktop/results/ex5/", "ex5", color="g")
