# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:24:11 2024

@author: Judith Recober
"""


import numpy as np
from scipy.spatial import Delaunay
import vtk
import pandas as pd
import sys
from PyQt5.QtWidgets import QFileDialog,QApplication
from PyQt5.QtCore import QDir



def choose_files():
    app = QApplication(sys.argv)
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.ExistingFile)  # Allow selecting only one existing file
    dialog.setNameFilter("CSV Files (*.csv)")
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Ensures consistent behavior
    dialog.setFilter(dialog.filter() | QDir.Hidden | QDir.AllDirs)  # Enable hidden files & directories

    if dialog.exec_():
        selected_files = dialog.selectedFiles()
        return selected_files[0] if selected_files else None

def plot_3d(senspos, senslabel, smooth_iterations=50, relaxation_factor=0.1, apply_smoothing=False):

    senspos = np.array(senspos, dtype=float)
    # Convert to polar coordinates for triangulation
    center_of_mass = np.mean(senspos, axis=0)
    senspos_centered = senspos - center_of_mass
    r = np.linalg.norm(senspos_centered, axis=1)
    theta = np.arccos(senspos_centered[:, 2] / r)
    phi = np.arctan2(senspos_centered[:, 0], senspos_centered[:, 1])
    x_proj = theta * np.cos(phi)
    y_proj = theta * np.sin(phi)
    polar_proj = np.vstack((x_proj, y_proj)).T

    # Delaunay triangulation
    tri = Delaunay(polar_proj)

    # Create VTK mesh from Delaunay triangulation
    points = vtk.vtkPoints()
    for pos in senspos:
        points.InsertNextPoint(pos[0], pos[1], pos[2])

    # Create Triangle Cells from Delaunay Triangulation
    triangles = vtk.vtkCellArray()
    for simplex in tri.simplices:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, simplex[0])
        triangle.GetPointIds().SetId(1, simplex[1])
        triangle.GetPointIds().SetId(2, simplex[2])
        triangles.InsertNextCell(triangle)

    # Create a PolyData object and assign points and polygons
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(triangles)
    
    
    if apply_smoothing:
        # Apply Laplacian Smoothing
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(poly_data)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        return smoother.GetOutput()
    else:
        # Return unsmoothed poly_data
        return poly_data





import vtk

def render_visualization(smoothed_poly_data, senslabel, empty_slots_aprox_positions):
    
    # VTK Renderer Setup
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    
    # Surface Mesh that we got with the Delaunay triangulation
    mesh_mapper = vtk.vtkPolyDataMapper() #mapper
    mesh_mapper.SetInputData(smoothed_poly_data)
    
    mesh_actor = vtk.vtkActor()#actor
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange color
    mesh_actor.GetProperty().SetOpacity(1) # fully opaque
    renderer.AddActor(mesh_actor)
    
    points = smoothed_poly_data.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    for i in range(num_points):
        # Get point coordinates
        x, y, z = points.GetPoint(i)
        
        # add the spheres with the points. These points are slots WITH sensors
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x, y, z) #centered in the corresponding position
        sphere_source.SetRadius(0.004)  # radius

        sphere_mapper = vtk.vtkPolyDataMapper() #mapper
        sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

        sphere_actor = vtk.vtkActor() #actor
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1, 1, 1)  
        renderer.AddActor(sphere_actor)
        
        #add the labels
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetInput(senslabel[i])
        text_actor.SetPosition(x , y, z) 
        text_actor.GetTextProperty().SetFontSize(30)
        text_actor.GetTextProperty().SetColor(1, 1, 1)  #
        renderer.AddActor(text_actor)
    
    # add the red spheres and text for empty slots
    for key in empty_slots_aprox_positions:
        [x2,y2,z2] = empty_slots_aprox_positions[key][-1] 
        
        sphere_source2 = vtk.vtkSphereSource()
        sphere_source2.SetCenter(x2, y2, z2) #centered in the corresponding position
        sphere_source2.SetRadius(0.005)  # radius

        sphere_mapper2 = vtk.vtkPolyDataMapper() #mapper
        sphere_mapper2.SetInputConnection(sphere_source2.GetOutputPort())

        sphere_actor2 = vtk.vtkActor() #actor
        sphere_actor2.SetMapper(sphere_mapper2)
        sphere_actor2.GetProperty().SetColor(1, 0, 0)  
        renderer.AddActor(sphere_actor2)
        
        #add the labels
        text_actor2 = vtk.vtkBillboardTextActor3D()
        text_actor2.SetInput(key)
        text_actor2.SetPosition(x2 , y2, z2) 
        text_actor2.GetTextProperty().SetFontSize(32)
        text_actor2.GetTextProperty().SetColor(1, 0, 0) 
        text_actor2.GetTextProperty().BoldOn()
        renderer.AddActor(text_actor2)
        
    
    
    # Mouse Rotation
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)
    
    # Set Up Camera and Window
    renderer.SetBackground(0,0,0)  # black background
    renderWindow.SetSize(800, 600)
    renderWindow.Render()
    
    # Start Interaction
    interactor.Start()

    

"""
In total there are 144 sensor slots

"""




def check_empty_slots(helmet_slots_template, input_data):
    
    slots_with_sensor = input_data["Coil Name"]
    empty_slots = list(set(helmet_slots_template.str[:-3]) - set(slots_with_sensor))
    
    return empty_slots # NO _bz in the names
    





def find_neighbours(empty_slots, helmet_slots_template, radius=1):

    
    outer_slots1 = ["101","201","301","401","501","601"]
    outer_slots2 = ["607","509","411", "313","215","117"]
    prefixes = ["R", "L"]
    

    neighbors = {}

    # Filter valid slots (non-empty slots)
    valid_slots = helmet_slots_template[
        ~helmet_slots_template['Labels'].isin(empty_slots)
    ].copy()
    

    for _, empty_row in helmet_slots_template[
            helmet_slots_template['Labels'].isin(empty_slots)
        ].iterrows():
        current_label = empty_row['Labels']
        
        
        condition1 = np.array([
            any((f"{prefix}{slot}" == current_label) for prefix in prefixes)
            for slot in outer_slots1
        ])
        
        condition2 = np.array([
            any((f"{prefix}{slot}" == current_label) for prefix in prefixes)
            for slot in outer_slots2
        ])
        
        if condition1.any():
            suffix = 2  # Start with "2"
            while True:
                closest_neighbors = current_label[:-1] + str(suffix)
                if closest_neighbors in empty_slots:
                    suffix += 1
                else:
                    break
                
            
        elif condition2.any():
            suffix = int(current_label[-1]) - 1
            while suffix >= 0:  # Ensure suffix doesn't go below 0
                closest_neighbors = current_label[:-1] + str(suffix)
                if closest_neighbors in empty_slots:
                    suffix -= 1
                else:
                    break
        
        else:
        
            current_x, current_y = empty_row['X_Pos'], empty_row['Y_Pos']
    
            # Calculate distances to all valid slots
            valid_slots['Distance'] = np.sqrt(
                (valid_slots['X_Pos'] - current_x) ** 2 +
                (valid_slots['Y_Pos'] - current_y) ** 2
            )
    
            # Filter slots within the radius and sort by distance
            nearby_slots = valid_slots[valid_slots['Distance'] <= radius].sort_values('Distance')
    
            # Extract up to 4 closest neighbor labels
            closest_neighbors = list(nearby_slots['Labels'][:4])
           
        # Assign neighbors to the current empty slot
        neighbors[current_label] = closest_neighbors

    return neighbors


def compute_positions(sensors_info, slots_with_neighbours):
    
    for key in slots_with_neighbours:
        neighbours = slots_with_neighbours[key]
        if isinstance(neighbours, str): # cases where here is just 1 neighbour (basically the outer slots)
            row = sensors_info[sensors_info[:, 0] == neighbours]
            x, y, z = row[0][1], row[0][2], row[0][3]
            new_z = z-0.02
            pos_empty_slot = [x,y,new_z]
            slots_with_neighbours[key] = [slots_with_neighbours[key]]
            slots_with_neighbours[key].append(pos_empty_slot)
            
        else: # cases where we have 4 neighbours (the rest)
           neighbours = slots_with_neighbours[key] 
           sum_x, sum_y, sum_z = 0, 0, 0
           for neighb in neighbours:
               #print (neighb)
               row = sensors_info[sensors_info[:, 0] == neighb]
               #print(row)
               x, y, z = row[0][1], row[0][2], row[0][3]
               sum_x += x
               sum_y += y
               sum_z += z
            
           mean_x = sum_x / len(neighbours)
           mean_y = sum_y / len(neighbours)
           mean_z = sum_z / len(neighbours)
           pos_empty_slot = [mean_x,mean_y,mean_z]
           slots_with_neighbours[key].append(pos_empty_slot)
    
    return slots_with_neighbours






if __name__ == "__main__":
    # Get all the available labels of the slots of the helmet
    helmet_slots_info = pd.read_csv('data_sensors.csv')
    helmet_slots_labels = helmet_slots_info['Labels']
    helmet_slots_template = helmet_slots_info[['Labels', 'X_Pos', 'Y_Pos']].copy()
    helmet_slots_template['Labels'] = helmet_slots_template['Labels'].str[:-3]
    
    
    # get the current input CSV file
    input_data_path = choose_files()
    input_data = pd.read_csv(input_data_path)
    
    # check, from this current CSV file, which slots are empty
    empty_slots =  check_empty_slots(helmet_slots_labels, input_data)
    
    # compute the mesh with the positions of the sensors that are inside the NON empty slots
    senspos = np.vstack((input_data["sensor_x"], input_data["sensor_y"], input_data["sensor_z"])).T
    senslabel= np.array(input_data["Channel"].str[:-3])
    smoothed_poly_data = plot_3d(senspos, senslabel)
    
    # get 4 neighbours per each empty slot
    empty_slots_with_neighbours = find_neighbours (empty_slots, helmet_slots_template)
    
    slots_info =  np.vstack((input_data["Coil Name"], input_data["sensor_x"], input_data["sensor_y"], input_data["sensor_z"])).T
    empty_slots_aprox_positions = compute_positions(slots_info, empty_slots_with_neighbours)
    
    
    # VTK rendering
    render_visualization(smoothed_poly_data, senslabel, empty_slots_aprox_positions)





