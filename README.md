# Mechanics-of-Composite-Material-CLT-python-OOP

## Overview
A comprehensive Python package for stress-strain analysis of laminated composite materials using Classical Laminate Theory (CLT). The package provides:

- **Ply-level analysis**: Material properties transformation for single plies
- **Laminate analysis**: ABD matrix calculation for multi-layered composites
- **Visualization**: Stress/strain distribution through thickness
- **Material database**: Predefined common composite materials

## Core Components

### 1. PlateComposite Class
Models a single composite lamina (unidirectional ply)

**Key capabilities:**
- Calculate compliance/stiffness matrices in material coordinates
- Transform properties to arbitrary coordinate systems
- Compute stress-strain relationships using plane stress assumption
- Handle both stress-to-strain and strain-to-stress conversions

**Example:**
```python
# Create a graphite-epoxy ply at 45°
ply = PlateComposite(graphite_epoxy, 45)

# Get transformed stiffness matrix
Q_bar = ply.Stiffness

# Calculate strains for applied stress
stress = [100e6, 50e6, 10e6]  # [σxx, σyy, τxy] in Pa
strain = ply.applyStress(stress)
