"""
Composite Mechanics Analyzer
A Python package for stress-strain analysis of laminated composites
using Classical Laminate Theory (CLT).
"""

import numpy as np
import matplotlib.pyplot as plt

# Material properties database

graphite_epoxy = [181e9, 10.30e9, 0.28, 7.17e9]  # [E1, E2, v12, G12] in Pa
glass_epoxy = [38.6e9, 8.27e9, 0.26, 4.14e9]     # [E1, E2, v12, G12] in Pa


def plot_curves(x_list, y_list, title, type):
	"""
	Plots strain/stress distribution through laminate thickness
    
	Args:
		x_list (list): List of z-coordinate arrays for each lamina
		y_list (list): List of strain/stress component arrays
		title (str): Plot title and figure name
		type (str): Plot type ("U" for strain, "S" for stress)
    
	Returns:
		matplotlib.figure: Generated figure object
	"""
	if type == "U":
		fig, ax = plt.subplots(num=title)
		for i, (x, y) in enumerate(zip(x_list, y_list), start=1):
			ax.plot(x, y, label=f'lamina_{i}')
		ax.legend()

	elif type == "S":
		fig, ax = plt.subplots(num=title)
		for i, (x, y) in enumerate(zip(x_list, y_list), start=1):
			ax.plot(x[1:-1], y, label=f'lamina_{i}')
		ax.legend()

	return fig


class PlateComposite:
	"""
	Represents a single composite lamina (unidirectional ply)
    
	This class models the mechanical behavior of a single composite layer
	with unidirectional fibers. It calculates:
	- Material properties in principal coordinates (E1, E2, v12, G12)
	- Transformed properties in arbitrary coordinate systems
	- Stress-strain relationships using plane stress assumption
    
	Key functionality:
	- Calculate compliance/stiffness matrices in material/global coordinates
	- Apply stress/strain transformations
	- Compute strains for given stresses and vice versa
    
	Args:
		material (list): [E1, E2, v12, G12] in Pa
		direction (float): Fiber orientation angle (degrees)
	"""

	def __init__(self, material, direction):
		self.E1 = material[0]
		self.E2 = material[1]
		self.v12 = material[2]
		self.G12 = material[3]
		self.angle = np.deg2rad(direction)


	@property
	def Compliance0(self):
		s11 = 1 / self.E1
		s12 = (-1 * self.v12) / self.E1
		s22 = 1 / self.E2
		s33 = 1 / self.G12

		S = np.array([[s11, s12, 0],
				[s12, s22, 0],
				[0, 0, s33]])

		return S


	@property
	def Stiffness0(self):
		v21 = (self.E2 * self.v12) / self.E1

		q11 = self.E1 / (1 - (self.v12 * v21))
		q12 = (self.v12 * self.E2) / (1 - (self.v12 * v21))
		q22 = self.E2 / (1 - (self.v12 * v21))
		q33 = self.G12

		Q = np.array([[q11, q12, 0],
				[q12, q22, 0],
				[0, 0, q33]])

		return Q


	@property
	def TransverseInv(self):
		m = np.cos(self.angle)
		l = np.sin(self.angle)

		T11 = m ** 2
		T12 = l ** 2
		T13 = -2 * m * l
		T21 = l ** 2
		T22 = m ** 2
		T23 = 2 * m * l
		T31 = m * l
		T32 = -1 * m * l
		T33 = (m ** 2) - (l ** 2)

		T = np.array([[T11, T12, T13],
				[T21, T22, T23],
				[T31, T32, T33]])

		return T


	@property
	def Transverse(self):
		R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
		Rinv = np.linalg.inv(R)

		m = np.cos(self.angle)
		l = np.sin(self.angle)

		T11 = m ** 2
		T12 = l ** 2
		T13 = 2 * m * l
		T21 = l ** 2
		T22 = m ** 2
		T23 = -2 * m * l
		T31 = -1 * m * l
		T32 = m * l
		T33 = (m ** 2) - (l ** 2)

		T = np.array([[T11, T12, T13],
				[T21, T22, T23],
				[T31, T32, T33]])

		return R @ T @ Rinv


	@property
	def Compliance(self):
		S_bar = self.TransverseInv @ self.Compliance0 @ self.Transverse
		return S_bar


	@property
	def Stiffness(self):
		Q_bar = self.TransverseInv @ self.Stiffness0 @ self.Transverse
		return Q_bar


	def get_stiffness(self):
		return self.Stiffness


	def applyStress(self, stress_matrix):
		stress_matrix = np.array(stress_matrix)
		Strain = self.Compliance @ stress_matrix
		return Strain


	def appllyStrain(self, strain_matrix):
		strain_matrix = np.array(strain_matrix)
		Stress = self.Stiffness @ strain_matrix
		return Stress


class LaminatedComposite:
	"""
	Models a multi-layered composite laminate using Classical Laminate Theory
    
	This class analyzes the mechanical behavior of layered composite structures
	consisting of multiple plies with different orientations. It implements:
	- ABD matrix calculation for laminate stiffness
	- Stress/strain recovery through thickness
	- Visualization of stress/strain distributions
    
	Key functionality:
	- Compute ABD matrix (extensional, coupling, bending stiffnesses)
	- Apply force/moment resultants or mid-plane strains/curvatures
	- Calculate stress/strain at any point through thickness
	- Generate plots of stress/strain distributions
    
	Args:
		laminate (list): List of tuples (material, angle, thickness)
			material = [E1, E2, v12, G12]
			angle = fiber orientation (degrees)
			thickness = ply thickness (m)
	"""

	def __init__(self, laminate):

		self.laminate = laminate
		self.laminate_number = len(laminate)

		self.Strain = np.zeros((6, 1))
		self.Stress = np.zeros((6, 1))


	@property
	def stiffness0(self):
		laminate_stiffness = []

		for i in range(self.laminate_number):
			lamina = PlateComposite(self.laminate[i][0], self.laminate[i][1])
			laminate_stiffness.append(lamina.get_stiffness())

		return laminate_stiffness


	@property
	def z(self):
		total_thickness = sum(lam[2] for lam in self.laminate)
		z = [-total_thickness / 2]
		current_z = -total_thickness / 2

		for i in self.laminate:
			current_z += i[2]
			z.append(round(current_z, 12))

		return z


	@property
	def ABDmatrix(self):
		A = np.zeros((3, 3))
		B = np.zeros((3, 3))
		D = np.zeros((3, 3))

		for n in range(self.laminate_number):
			Q = self.stiffness0[n]
			z_n = self.z[n]
			z_n1 = self.z[n + 1]
			thickness = z_n1 - z_n

			A += Q * thickness
			B += Q * (z_n1 ** 2 - z_n ** 2) / 2
			D += Q * (z_n1 ** 3 - z_n ** 3) / 3

		return np.block([[A, B], [B, D]])


	def getABDmatrix(self):
		return self.ABDmatrix


	def applyStress(self, Stress_):
		"""Stress = [N_xx, N_yy, N_xy, M_xx, M_yy, M_xy]"""

		compliance = np.linalg.inv(self.ABDmatrix)
		self.Strain = compliance @ Stress_
		self.Stress = Stress_


	def applyStrain(self, Strain_):
		"""Strain = [u_xx, u_yy, u_xy, k_xx, k_yy, k_xy]"""

		self.Stress = self.ABDmatrix @ Strain_
		self.Strain = Strain_


	def getStrainPlane(self, z_):
		return self.Strain[:3] + (z_ * self.Strain[3:])


	def getStressPlane(self, z_):
		StrainPlane = self.Strain[:3] + (z_ * self.Strain[3:])

		for i in range(self.laminate_number):
			if z_ < self.z[i + 1] and z_ > self.z[i]:
				Q = self.stiffness0[i]
			elif z_ == self.z[i] or z_ == self.z[i + 1]:
				return None

		return Q @ StrainPlane


	def plotStrainPlane(self, U):
		"""U = [U1, U2, U3]"""
		Z = []
		U_ = []
		
		for n in range(self.laminate_number):
			z_n = self.z[n]
			z_n1 = self.z[n + 1]

			Z.append(np.linspace(z_n, z_n1, 100))
		
		for z in Z:
			U_.append([])
			for i in z:
				m = self.getStrainPlane(i)
				if type(m) == type(np.array([])):
					U_[-1].append(m)

		U1, U2, U3 = [], [], []

		for i in U_:
			U1.append([])
			U2.append([])
			U3.append([])
			for j in i:
				u1, u2, u3 = j

				U1[-1].append(u1)
				U2[-1].append(u2)
				U3[-1].append(u3)

		plot_curves(Z, U1, "U1", "U") if U[0] else ...
		plot_curves(Z, U2, "U2", "U") if U[1] else ...
		plot_curves(Z, U3, "U3", "U") if U[2] else ...

		plt.show()


	def plotStressPlane(self, S):
		"""S = [S1, S2, S3]"""
		Z = []
		S_ = []
		
		for n in range(self.laminate_number):
			z_n = self.z[n]
			z_n1 = self.z[n + 1]

			Z.append(np.linspace(z_n, z_n1, 100))
		
		for z in Z:
			S_.append([])
			for i in z:
				m = self.getStressPlane(i)
				if type(m) == type(np.array([])):
					S_[-1].append(m)

		S1, S2, S3 = [], [], []

		for i in S_:
			S1.append([])
			S2.append([])
			S3.append([])
			for j in i:
				s1, s2, s3 = j

				S1[-1].append(s1)
				S2[-1].append(s2)
				S3[-1].append(s3)

		if S[0]:
			plot_curves(Z, S1, "S1", "S")
		if S[1]:
			plot_curves(Z, S2, "S2", "S")
		if S[2]:
			plot_curves(Z, S3, "S3", "S")

		plt.show()
