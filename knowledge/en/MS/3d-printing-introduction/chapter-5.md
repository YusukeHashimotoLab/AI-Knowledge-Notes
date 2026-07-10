---
title: "Chapter 5: Python in Practice - 3D Printing Simulation"
chapter_title: "Chapter 5: Python in Practice - 3D Printing Simulation"
subtitle: Running STL analysis, slicing, heat conduction, parameter optimization, and machine learning in code
reading_time: 45-50 minutes
difficulty: Intermediate to Advanced
code_examples: 5
exercises: 5
---

[AI Terakoya Home](<../index.html>) › [Materials Science](<../../index.html>) › [Advanced Materials Systems](<../../MS/advanced-materials-systems-introduction/index.html>) › Chapter 5

🌐 EN | [🇯🇵 JP](<../../../jp/MS/3d-printing-introduction/chapter-5.html>) | Last sync: 2025-11-16

## Learning Objectives

After completing this chapter, you will be able to explain and practice the following:

### Basic Understanding (Level 1)

  * Understand that an STL (STereoLithography) file represents an object's surface as a triangle mesh, and explain the relationship between vertices, normals, and edges
  * Explain that slicing is "a geometric operation that finds the intersection of a plane with each triangle, layer by layer"
  * Understand that temperature change during printing can be described by the transient heat conduction equation
  * Explain why process parameter optimization and machine learning are useful in 3D printing

### Practical Skills (Level 2)

  * Generate a triangle mesh with NumPy alone and compute its volume, surface area, normals, and watertightness
  * Implement a plane-intersection algorithm to obtain the cross-section contour and its perimeter at any height
  * Solve the 1D transient heat conduction equation with an explicit finite difference scheme and draw a cooling curve
  * Quantify the trade-off between build time and surface quality using grid search

### Applied Ability (Level 3)

  * Judge printability from mesh quality metrics (watertightness, normal consistency)
  * Connect thermal simulation results to the discussion of interlayer bonding and warping (Chapter 2)
  * Train a regression model on synthetic data and read the dominant parameters from feature importances
  * Understand the assumptions and limits of each method, and judge how far to trust it in practice

**💡 Where This Chapter Fits**

Chapters 1 through 4 systematically covered the principles, materials, and processes of additive manufacturing (AM). As the capstone of the series, this chapter turns that knowledge into **working Python code**. We work hands-on through five practical tasks in order: loading STL, slicing, thermal simulation, parameter optimization, and machine learning. Every code example here was run with `python3`, and the outputs shown are the actual execution results. We limit external libraries to NumPy and scikit-learn, and implement STL processing ourselves rather than relying on a dedicated library, so that the inner workings remain fully visible.

## 5.1 Loading and Analyzing STL Files

### 5.1.1 Representing an Object as a Triangle Mesh

An STL (STereoLithography) file represents an object's surface as a set of **triangle meshes**. Each triangle has three vertex coordinates and one outward normal vector. Here we analyze this format—whose structure we learned in Chapter 1—with our own code. By building the triangle array with NumPy alone, without a dedicated library such as numpy-stl, we make clear "from which formula" the volume and surface area are obtained.

As the analysis target, we generate a cube with a 20 mm edge as 12 triangles (2 per face). Because a cube has known analytic values for volume and surface area (20³ = 8000 mm³ and 6×20² = 2400 mm² respectively), it is ideal for verifying the implementation.

### 5.1.2 Computing Volume, Surface Area, and Watertightness

Three formulas are fundamental for obtaining geometric quantities from a triangle mesh.

  * **Surface area** : the sum of each triangle's area ½|(v₂−v₁)×(v₃−v₁)|
  * **Volume** : the absolute value of the sum of signed tetrahedra (v₁·(v₂×v₃))/6 formed by the origin and each triangle (a divergence-theorem-based method)
  * **Watertightness** : every undirected edge is shared by exactly two triangles. If an edge is shared by only one, there is a hole

#### Code Example 1: STL Mesh Analysis with NumPy
    
    
    import numpy as np
    
    def make_cube_mesh(size=20.0):
        """Generate a cube of edge `size` as 12 triangles (outward normals)."""
        s = size
        v = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],   # bottom z=0
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],   # top z=s
        ], dtype=float)
        # Each face defined counter-clockwise (CCW seen from outside)
        faces = [
            (0, 3, 2), (0, 2, 1),   # bottom (-z)
            (4, 5, 6), (4, 6, 7),   # top (+z)
            (0, 1, 5), (0, 5, 4),   # front (-y)
            (2, 3, 7), (2, 7, 6),   # back (+y)
            (1, 2, 6), (1, 6, 5),   # right (+x)
            (0, 4, 7), (0, 7, 3),   # left (-x)
        ]
        tris = np.array([[v[a], v[b], v[c]] for a, b, c in faces])
        return tris
    
    def triangle_normals(tris):
        """Unit normal of each triangle: (v2-v1)x(v3-v1)."""
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        n = np.cross(e1, e2)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return n / lengths
    
    def surface_area(tris):
        """Sum of triangle areas = surface area."""
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    
    def mesh_volume(tris):
        """Volume by the signed-tetrahedron method (origin + each triangle)."""
        v1, v2, v3 = tris[:, 0], tris[:, 1], tris[:, 2]
        signed = np.einsum('ij,ij->i', v1, np.cross(v2, v3)) / 6.0
        return abs(signed.sum())
    
    def is_watertight(tris, tol=6):
        """Check that every undirected edge is shared by exactly 2 triangles."""
        verts = np.round(tris.reshape(-1, 3), tol)
        uniq, inv = np.unique(verts, axis=0, return_inverse=True)
        idx = inv.reshape(-1, 3)
        edge_count = {}
        for a, b, c in idx:
            for u, w in [(a, b), (b, c), (c, a)]:
                key = (min(u, w), max(u, w))
                edge_count[key] = edge_count.get(key, 0) + 1
        counts = np.array(list(edge_count.values()))
        watertight = bool(np.all(counts == 2))
        return watertight, len(uniq), len(edge_count), counts
    
    mesh = make_cube_mesh(20.0)
    normals = triangle_normals(mesh)
    area = surface_area(mesh)
    vol = mesh_volume(mesh)
    wt, n_vertices, n_edges, counts = is_watertight(mesh)
    bbox_min = mesh.reshape(-1, 3).min(axis=0)
    bbox_max = mesh.reshape(-1, 3).max(axis=0)
    
    print(f"Triangles       : {len(mesh)}")
    print(f"Unique vertices : {n_vertices}")
    print(f"Unique edges    : {n_edges}")
    print(f"Surface area    : {area:.2f} mm^2  (analytic 6*20^2 = {6*20**2})")
    print(f"Volume          : {vol:.2f} mm^3  (analytic 20^3 = {20**3})")
    print(f"Bounding box    : min={bbox_min}, max={bbox_max}")
    print(f"Watertight      : {wt}  (edge-share count min={counts.min()}, max={counts.max()})")
    print(f"First 3 unit normals:")
    for i in range(3):
        print(f"  face {i}: [{normals[i,0]:+.2f} {normals[i,1]:+.2f} {normals[i,2]:+.2f}]")
    

**Execution result:**
    
    
    Triangles       : 12
    Unique vertices : 8
    Unique edges    : 18
    Surface area    : 2400.00 mm^2  (analytic 6*20^2 = 2400)
    Volume          : 8000.00 mm^3  (analytic 20^3 = 8000)
    Bounding box    : min=[0. 0. 0.], max=[20. 20. 20.]
    Watertight      : True  (edge-share count min=2, max=2)
    First 3 unit normals:
      face 0: [+0.00 +0.00 -1.00]
      face 1: [+0.00 +0.00 -1.00]
      face 2: [+0.00 +0.00 +1.00]
    

The volume (8000 mm³) and surface area (2400 mm²) match the analytic values exactly, confirming the implementation is correct. The 18 unique edges and every edge being shared exactly twice mean this mesh is **watertight (a closed surface)**. The counts 8 vertices, 18 edges, and 12 faces also satisfy Euler's polyhedron formula V−E+F = 8−18+12 = 2. That all normals point outward (bottom −z, top +z) is likewise the condition for correct inside/outside determination during printing.

**⚠️ Assumptions of the Watertightness Check**

The edge-share check here treats vertices as identical by rounding their coordinates to six decimal places. In real STL files, "duplicate vertices"—where vertices at the same location carry slightly different floating-point values when exported from CAD—occur frequently. If the rounding precision is wrong, edges that are actually connected are judged distinct, and a healthy mesh is falsely flagged as having holes. In practice, libraries such as trimesh merge vertices with a tolerance before judging. We keep the implementation minimal here to make the mechanism understandable.

## 5.2 Implementing a Slicing Algorithm

### 5.2.1 Plane-Triangle Intersection

**Slicing** is the process of cutting a 3D model with a horizontal plane at each fixed height (layer height) and extracting the contour of each layer (see Chapter 1). Its core is the geometric operation of **"finding the intersection segment of a triangle with the plane z = z₀."** Of a triangle's three edges, only those whose endpoints lie on opposite sides of the plane cross it. The intersection is found by linear interpolation between the endpoints, and exactly two intersection points—that is, one segment—are obtained from a single triangle.

intersection = p₀ + t·(p₁ − p₀), t = (z − z_p₀) / (z_p₁ − z_p₀) 

Collecting these segments at each layer yields that layer's cross-section contour. Here, as an example whose cross-section changes with height, we slice a square pyramid with a 20 mm base and 30 mm height. The pyramid's cross-section is a square that shrinks as the height increases, so we can confirm the contour extraction works correctly from the decreasing perimeter.

#### Code Example 2: Slicing by Plane Intersection
    
    
    import numpy as np
    
    def make_pyramid_mesh(base=20.0, height=30.0):
        """Square pyramid mesh (4 side + 2 base triangles)."""
        b, h = base, height
        apex = [b/2, b/2, h]
        v = np.array([
            [0, 0, 0], [b, 0, 0], [b, b, 0], [0, b, 0], apex,
        ], dtype=float)
        faces = [
            (0, 2, 1), (0, 3, 2),        # base
            (0, 1, 4), (1, 2, 4),        # sides
            (2, 3, 4), (3, 0, 4),
        ]
        return np.array([[v[a], v[b], v[c]] for a, b, c in faces])
    
    def slice_at_z(tris, z):
        """Intersect each triangle with plane z=z0, return contour perimeter."""
        segments = []
        for tri in tris:
            pts = []
            for i in range(3):
                p0, p1 = tri[i], tri[(i + 1) % 3]
                z0, z1 = p0[2], p1[2]
                if (z0 - z) * (z1 - z) < 0:          # edge crosses the plane
                    t = (z - z0) / (z1 - z0)
                    pts.append(p0 + t * (p1 - p0))
            if len(pts) == 2:
                segments.append((pts[0], pts[1]))
        perimeter = sum(np.linalg.norm(a - b) for a, b in segments)
        return segments, perimeter
    
    pyr = make_pyramid_mesh(base=20.0, height=30.0)
    layer_height = 0.2
    z_max = 30.0
    n_layers = int(z_max / layer_height)
    print(f"Model: square pyramid base 20mm height 30mm, layer height {layer_height} mm")
    print(f"Total layers: {n_layers}")
    print(f"{'z (mm)':>8} {'segments':>9} {'perimeter (mm)':>15}")
    total_path = 0.0
    for z in [1.0, 6.0, 12.0, 18.0, 24.0, 29.0]:
        segs, perim = slice_at_z(pyr, z)
        print(f"{z:8.1f} {len(segs):9d} {perim:15.3f}")
    
    # Accumulate all-layer contour path length (used to estimate print time)
    for i in range(1, n_layers + 1):
        z = i * layer_height
        _, perim = slice_at_z(pyr, min(z, z_max - 1e-6))
        total_path += perim
    print_speed = 50.0  # mm/s
    print(f"Total contour path length: {total_path:.1f} mm")
    print(f"Perimeter-only print time estimate ({print_speed:.0f} mm/s): {total_path/print_speed:.1f} s")
    

**Execution result:**
    
    
    Model: square pyramid base 20mm height 30mm, layer height 0.2 mm
    Total layers: 150
      z (mm)  segments  perimeter (mm)
         1.0         4          77.333
         6.0         4          64.000
        12.0         4          48.000
        18.0         4          32.000
        24.0         4          16.000
        29.0         4           2.667
    Total contour path length: 5960.0 mm
    Perimeter-only print time estimate (50 mm/s): 119.2 s
    

Exactly four segments (the four sides of the square) are obtained at each layer, and the perimeter decreases linearly with height. At z = 1 mm the perimeter is 77.3 mm, and near the apex at z = 29 mm it is 2.67 mm, correctly capturing the similar shrinkage of the pyramid. Summing the contour path length over all 150 layers gives about 5960 mm, yielding an estimate of about 119 seconds if only the perimeter were drawn at 50 mm/s. In real printing, infill, travel moves, and acceleration/deceleration are added, so this is only a lower-bound estimate.

**💡 The Scope of This Implementation vs. Practice**

Here we implemented up to "collecting segments and measuring the perimeter." A practical slicer connects these segments at their endpoints to order them into closed contours (loops), determines inside/outside to generate shells and infill, and further converts them to G-code (Chapter 1). In particular, handling cross-sections with multiple holes or self-intersecting contours, and connecting segments whose endpoints are slightly offset by floating-point error, are the hard parts of implementation. This code shows the geometric core; it does not by itself constitute a production slicer.

## 5.3 Thermal Simulation: Cooling of a Layer

### 5.3.1 The Transient Heat Conduction Equation

In Chapter 2 we learned that interlayer bonding "proceeds only while the interface is above the glass transition temperature (Tg)." So how fast does an extruded layer actually cool? The quantity that handles this is the **transient heat conduction equation**. In one dimension it takes the following form.

∂T/∂t = α · ∂²T/∂x², α = k / (ρ·c) (thermal diffusivity) 

Here α is the **thermal diffusivity** , determined from the thermal conductivity k, density ρ, and specific heat c. We solve this partial differential equation with the **explicit finite difference method** , which approximates it by dividing space into a grid and using finite differences. The explicit method is clear to implement, but if the time step dt is too large it diverges numerically, so we must respect the **stability condition** that the grid Fourier number r = α·dt/dx² be at most 0.5.

Here we compute the process in which a 0.8 mm thin wall, just extruded, solidifies while cooling by convection from both faces into the surrounding air. The boundary condition incorporates convection (Newton's law of cooling) via the ghost-node method.

#### Code Example 3: Explicit Solution of 1D Transient Heat Conduction
    
    
    import numpy as np
    
    # A freshly extruded thin wall cools by convection from both faces while
    # solidifying. Solve 1D transient conduction dT/dt = alpha d2T/dx2 explicitly.
    L = 0.8e-3        # wall thickness m (0.8 mm)
    nx = 21           # grid points
    dx = L / (nx - 1)
    alpha = 1.3e-7    # PLA thermal diffusivity m^2/s
    T_ext = 210.0     # extrusion temperature C
    T_env = 30.0      # ambient temperature C
    Tg = 60.0         # PLA glass transition temperature C
    h = 40.0          # convective heat transfer coefficient W/(m^2 K)
    k = 0.13          # PLA thermal conductivity W/(m K)
    
    dt = 0.2 * dx**2 / alpha          # time step meeting stability (Fourier <= 0.5)
    r = alpha * dt / dx**2
    Bi = h * dx / k                   # grid Biot number
    print(f"Grid spacing dx = {dx*1e6:.1f} um, time step dt = {dt*1000:.3f} ms")
    print(f"Grid Fourier r = {r:.3f} (<=0.5 stable), grid Biot Bi = {Bi:.4f}")
    
    T = np.full(nx, T_ext)
    t = 0.0
    t_center_Tg = None
    checkpoints = [0.0, 0.5, 1.0, 2.0, 5.0]
    records = {}
    next_cp = 0
    max_steps = 2_000_000
    for step in range(max_steps):
        Tn = T.copy()
        # Interior nodes: explicit finite difference
        Tn[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
        # Convective boundaries (both ends) via ghost-node method
        Tn[0] = T[0] + 2*r*(T[1] - T[0]) - 2*r*Bi*(T[0] - T_env)
        Tn[-1] = T[-1] + 2*r*(T[-2] - T[-1]) - 2*r*Bi*(T[-1] - T_env)
        T = Tn
        t += dt
        if t_center_Tg is None and T[nx//2] <= Tg:
            t_center_Tg = t
        while next_cp < len(checkpoints) and t >= checkpoints[next_cp]:
            records[checkpoints[next_cp]] = (T[0], T[nx//2])
            next_cp += 1
        if T.max() <= Tg and t_center_Tg is not None:
            break
    
    print(f"{'time (s)':>9} {'surface (C)':>13} {'center (C)':>12}")
    for cp in checkpoints:
        if cp in records:
            surf, cen = records[cp]
            print(f"{cp:9.1f} {surf:13.1f} {cen:12.1f}")
    print(f"Time for center to reach Tg={Tg:.0f}C: {t_center_Tg:.2f} s")
    

**Execution result:**
    
    
    Grid spacing dx = 40.0 um, time step dt = 2.462 ms
    Grid Fourier r = 0.200 (<=0.5 stable), grid Biot Bi = 0.0123
     time (s)   surface (C)   center (C)
          0.0         209.1        210.0
          0.5         194.7        204.8
          1.0         187.0        196.7
          2.0         172.6        181.5
          5.0         136.9        143.5
    Time for center to reach Tg=60C: 18.86 s
    

The grid Fourier number r = 0.20 satisfies the stability condition (≤0.5), so the computation does not diverge and gives a smooth cooling curve. Because the surface contacts the air, it cools slightly faster than the center, but the grid Biot number Bi = 0.012 is very small (i.e., internal conduction is fast enough relative to convection), so the surface-to-center temperature difference is small and the whole wall cools almost uniformly. It takes about 19 seconds for the center to drop below Tg (60°C)—this is because **the surroundings are relatively warm at 30°C** ; lowering the ambient temperature with a cooling fan would shorten this time greatly. The qualitative conclusion from Chapter 2 that "a warmer environment gives a longer welding window" is here backed up by a concrete number of seconds.

**⚠️ Simplification of the Model**

This model treats only 1D convective cooling, ignoring conduction to the layer below or the build plate, latent heat during extrusion, and temperature-dependent properties. It therefore tends to be slower than actual cooling. The aim is not precise prediction of absolute time, but to experience "that a cooling curve is obtained from an explicit method satisfying the stability condition" and "that ambient temperature and heat transfer govern the cooling rate." Quantitative design uses higher-order 2D/3D models or the finite element method.

## 5.4 Process Parameter Optimization

### 5.4.1 The Trade-off Between Build Time and Surface Quality

On the 3D printing shop floor, we must satisfy the conflicting demands of "as fast as possible, as clean as possible." These two often point in opposite directions. Increasing the layer height reduces the number of layers and shortens build time, but the layer lines stand out and the surface becomes rougher. Here we brute-force two parameters—layer height and print speed—with a **grid search** and quantify the trade-off.

We estimate build time and surface roughness with the following concise model.

  * **Build time** ∝ (part height / layer height) × (path per layer / print speed)
  * **Surface roughness Ra** ∝ the square of the layer height (a geometric model approximating the layer lines as triangles, Ra ≈ LH²/8)

We then search for the condition that minimizes build time among those satisfying the quality constraint "surface roughness Ra at most 6 µm."

#### Code Example 4: Parameter Optimization by Grid Search
    
    
    # Evaluate the build-time vs surface-quality trade-off by grid search.
    # Build time ~ height/(layer height) x path/(speed)
    # Surface roughness Ra ~ proportional to layer height (layer lines)
    part_height = 30.0      # mm
    path_per_layer = 80.0   # mm/layer (representative perimeter + infill)
    layer_heights = [0.10, 0.15, 0.20, 0.30]
    speeds = [40, 60, 80, 100]     # mm/s
    Ra_max = 6.0            # allowable surface roughness um
    
    print(f"{'LH(mm)':>7} {'v(mm/s)':>8} {'time(min)':>10} {'Ra(um)':>8} {'accept':>7}")
    results = []
    for lh in layer_heights:
        n_layers = part_height / lh
        Ra = 1000.0 * lh**2 / 8.0    # geometric layer-line model Ra ~ LH^2/8, in um
        for v in speeds:
            t_build = n_layers * path_per_layer / v / 60.0   # min
            ok = Ra <= Ra_max
            results.append((lh, v, t_build, Ra, ok))
            flag = "OK" if ok else "NG"
            print(f"{lh:7.2f} {v:8d} {t_build:10.2f} {Ra:8.2f} {flag:>7}")
    
    feasible = [r for r in results if r[4]]
    best = min(feasible, key=lambda x: x[2])
    print(f"\nShortest build time satisfying Ra<={Ra_max:.0f}um:")
    print(f"  layer height {best[0]:.2f} mm, speed {best[1]} mm/s, "
          f"build time {best[2]:.2f} min, Ra {best[3]:.2f} um")
    

**Execution result:**
    
    
     LH(mm)  v(mm/s)  time(min)   Ra(um)  accept
       0.10       40      10.00     1.25      OK
       0.10       60       6.67     1.25      OK
       0.10       80       5.00     1.25      OK
       0.10      100       4.00     1.25      OK
       0.15       40       6.67     2.81      OK
       0.15       60       4.44     2.81      OK
       0.15       80       3.33     2.81      OK
       0.15      100       2.67     2.81      OK
       0.20       40       5.00     5.00      OK
       0.20       60       3.33     5.00      OK
       0.20       80       2.50     5.00      OK
       0.20      100       2.00     5.00      OK
       0.30       40       3.33    11.25      NG
       0.30       60       2.22    11.25      NG
       0.30       80       1.67    11.25      NG
       0.30      100       1.33    11.25      NG
    
    Shortest build time satisfying Ra<=6um:
      layer height 0.20 mm, speed 100 mm/s, build time 2.00 min, Ra 5.00 um
    

A layer height of 0.30 mm gives Ra = 11.25 µm and fails the quality constraint (NG). Looking at build time alone, 0.30 mm at 100 mm/s is fastest at 1.33 minutes, but it sacrifices surface quality. The fastest within the quality constraint is **layer height 0.20 mm and speed 100 mm/s** , giving a build time of 2.00 minutes and Ra of 5.00 µm. When multiple objectives conflict like this, formulating it as "constrained optimization" makes the decision clear.

**💡 Where Grid Search Fits**

When there are two or three parameters, each with a few levels, a brute-force grid search is the clearest and most reliable. However, as the number of parameters grows, the number of combinations explodes exponentially (the curse of dimensionality). In practice, methods such as Bayesian optimization and genetic algorithms are used to find good solutions with fewer trials. The coefficients of the model shown here (Ra ≈ LH²/8, etc.) are simplifications for illustration; actual surface roughness also depends on nozzle shape, material, and cooling. Use it as a tool for grasping trends.

## 5.5 Machine-Learning Process Prediction

### 5.5.1 Linking Process and Quality with Data

The simulations so far were **physics-based models** that write physical laws (geometry, heat conduction) down as equations. On the other hand, phenomena such as interlayer bonding, defects, and final strength involve many factors intertwined in ways that are hard to write as a simple equation. This is where **machine learning** , which learns the relationship between parameters and results from experimental or simulation data, becomes useful.

Here we build a model that predicts **tensile strength** from four process parameters: nozzle temperature, layer height, print speed, and infill density. In place of experimental data, we use 500 points of **synthetic data** generated by a known function (reflecting the physical trends of Chapter 2: hotter improves welding and strength, thicker layers weaken interlayer bonding, faster reduces welding time and strength, and infill contributes almost linearly to strength) plus noise. We use **Random Forest Regression** as the regressor, evaluate accuracy with the coefficient of determination R² and the mean absolute error (MAE), and read the dominant parameters from the feature importances.

#### Code Example 5: Strength Prediction with a Random Forest
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    
    rng = np.random.default_rng(42)
    n = 500
    # Synthetic data: process parameters -> tensile strength (MPa)
    nozzle_T = rng.uniform(190, 230, n)     # nozzle temperature C
    layer_h  = rng.uniform(0.10, 0.30, n)   # layer height mm
    speed    = rng.uniform(40, 100, n)      # print speed mm/s
    infill   = rng.uniform(20, 100, n)      # infill density %
    
    # Known generating function (interlayer welding + infill) + noise
    strength = (
        18.0
        + 0.22 * (nozzle_T - 190)          # hotter -> better interlayer welding
        - 55.0 * (layer_h - 0.10)          # thicker layers -> weaker bonding
        - 0.05 * (speed - 40)              # faster -> less welding time, weaker
        + 0.28 * infill                    # infill contributes almost linearly
        + rng.normal(0, 2.0, n)
    )
    
    X = np.column_stack([nozzle_T, layer_h, speed, infill])
    y = strength
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    
    print(f"Training samples: {len(X_tr)}, test samples: {len(X_te)}")
    print(f"Test R^2 : {r2:.3f}")
    print(f"Test MAE : {mae:.2f} MPa")
    names = ["nozzle_temp", "layer_height", "print_speed", "infill"]
    print("Feature importances:")
    for nm, imp in sorted(zip(names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {nm:12s}: {imp:.3f}")
    
    # Predict strength for an unseen condition
    q = np.array([[220.0, 0.15, 50.0, 80.0]])
    pq = model.predict(q)[0]
    print(f"Prediction: nozzle 220C/LH 0.15mm/50mm/s/infill 80% -> tensile strength {pq:.1f} MPa")
    

**Execution result:**
    
    
    Training samples: 375, test samples: 125
    Test R^2 : 0.906
    Test MAE : 1.93 MPa
    Feature importances:
      infill      : 0.686
      layer_height: 0.159
      nozzle_temp : 0.123
      print_speed : 0.031
    Prediction: nozzle 220C/LH 0.15mm/50mm/s/infill 80% -> tensile strength 42.7 MPa
    

On the test data, the coefficient of determination R² = 0.906 and the mean absolute error is 1.93 MPa—good predictive accuracy even for unseen parameter combinations. Looking at the feature importances, **infill density (0.686)** is overwhelmingly dominant, followed by layer height and nozzle temperature, with print speed contributing little. This is consistent with the generating function, in which the infill coefficient (0.28 × up to 80) affects strength most, showing that the model correctly learned the physical trends. In practice, such an importance analysis provides guidance on "which parameters to manage and optimize as a priority."

**⚠️ What It Means That the Data Is Synthetic**

The data here is **synthetic** , generated from a known function, and the model is highly accurate because we trained it on a function whose "answer" we know. In real printing, material lot variation, machine-to-machine differences, and environmental fluctuations are added, and the same R² cannot be expected. Note also that predictions outside the range of the training data (extrapolation) are not reliable. Machine learning is not an omnipotent prediction device; it becomes effective **only when there is sufficient quality and quantity of data**. The aim here is to learn "how to build the pipeline" and "how to read the importances."

## Exercises

These exercises check your understanding. Try to think for yourself before opening the answers.

Exercise 1 (Basic): The Meaning of Watertightness

Suppose that in Code Example 1 you accidentally delete one triangle of the cube mesh. How does the result of `is_watertight` change, and what problem does that cause for printing?

Show answer

Deleting one triangle reduces from 2 to 1 the share count of those edges that this triangle had shared with adjacent triangles (of its three edges). Because edges with a share count of 1 arise, `np.all(counts == 2)` becomes False and it is judged **watertight = False** (has holes). A mesh with a hole in its surface has an ambiguous inside/outside distinction, so the slicer cannot close the cross-section contour, and printing fails or produces unintended voids or defects.

Exercise 2 (Calculation): Predicting the Slice Perimeter

If you slice the pyramid of Code Example 2 (base 20 mm, height 30 mm) at z = 15 mm, what are the side length and perimeter of the cross-section square? Compute by hand from the similarity ratio.

Show answer

Since the pyramid converges to the apex at a height of 30 mm, the side of the cross-section square at height z is 20 ×(1 − z/30). At z = 15 mm this is 20 ×(1 − 0.5) = **10 mm**. The perimeter is 4 × 10 = **40 mm**. This is consistent as the midpoint between the outputs of Code Example 2 (48 at z=12, 32 at z=18).

Exercise 3 (Discussion): The Stability Condition

In Code Example 3, if you change the time step to `dt = 0.6 * dx**2 / alpha`, what does the grid Fourier number r become, and what do you predict for the computation result?

Show answer

r = α·dt/dx² = 0.6, which **exceeds the explicit method's stability condition r ≤ 0.5**. In this case the numerical solution amplifies oscillations at each time step, and the temperature runs away to physically impossible values (e.g., divergence beyond hundreds of degrees or negative temperatures). In the explicit method the stability condition is absolute, and the finer dx becomes, the smaller dt must be. If you do not want to be bound by the stability condition, use the unconditionally stable implicit method.

Exercise 4 (Applied): Changing the Constraint

In Code Example 4, if you tighten the allowable surface roughness to `Ra_max = 3.0` (µm), how does the shortest-build-time condition change? Answer from the values in the table.

Show answer

Only layer heights 0.10 mm (Ra=1.25) and 0.15 mm (Ra=2.81) satisfy Ra ≤ 3.0 µm; 0.20 mm (Ra=5.00) is excluded. Within this range the shortest time is **layer height 0.15 mm and speed 100 mm/s** at 2.67 minutes. Tightening the quality requirement lowers the upper limit of usable layer height and lengthens build time—the trade-off appears clearly.

Exercise 5 (Applied): Interpreting Feature Importances

In Code Example 5, if "print speed" turned out to have the largest feature importance, what interpretations are possible about the generating function or the real data?

Show answer

Print-speed importance becomes largest when strength depends strongly on speed. In terms of the generating function, this is a situation where the speed coefficient (currently −0.05) is larger than the others, or where speed varies over a wide range so that its contribution to strength dominates. If this happened with real data, it would suggest that for that machine and material, **insufficient welding time due to higher speed (reduced interlayer bonding) is the bottleneck of strength**. Candidate countermeasures are revising the speed upper limit or raising the nozzle temperature (the interlayer strength model of Chapter 2). Importance is a diagnostic tool that tells us "which factor to manage as a priority."

## Summary

In this chapter, we hands-on confirmed the additive manufacturing knowledge learned throughout the series as five executable Python code examples. The key points are as follows.

  * **STL analysis (5.1)** : Compute volume, surface area, normals, and watertightness from a triangle mesh with NumPy alone. Match with analytic values verifies the implementation.
  * **Slicing (5.2)** : The geometric operation of "finding the intersection segment of a plane with a triangle" is the core. The contour perimeter lets us estimate the print path.
  * **Thermal simulation (5.3)** : Solve the transient heat conduction equation with an explicit method to obtain a cooling curve. Confirm the stability condition and that ambient temperature governs cooling.
  * **Parameter optimization (5.4)** : Quantify the build-time vs surface-quality trade-off by grid search, and select the optimum under a constraint.
  * **Machine learning (5.5)** : Train a random forest regression on synthetic data and identify the dominant parameters from feature importances.
  * **A common stance** : Every method has assumptions and limits. Do not over-trust absolute values; use them as trends and decision support.

**✅ Series Complete**

Congratulations. Across all five chapters, you have learned the principles of additive manufacturing (Chapter 1), material extrusion (Chapter 2), vat photopolymerization and powder bed fusion (Chapter 3), material jetting, binder jetting, and more (Chapter 4), and the Python practice of this chapter. You should now have both an understanding of the principles and the ability to verify them in code. From here, work with real 3D printers and open-source slicers and mesh libraries (such as trimesh), and apply them to your own problems.

## Next Steps

The code covered in this chapter deliberately minimizes external libraries to lay the mechanisms bare. As extensions toward practice and research, there are directions such as the following.

  * **Full-scale mesh processing** : Use trimesh or Open3D to experience loading, repairing, Boolean operations, and visualization of real STL files.
  * **Going deeper into slicers** : Implement contour-loop ordering, infill generation, and G-code output, and compare with the output of Cura/PrusaSlicer.
  * **Advancing heat and stress** : Use 2D/3D finite element methods (FEM) to quantitatively simulate warping and residual stress (Chapter 2).
  * **Applying machine learning to real data** : Collect data you print and test yourself, and use Bayesian optimization to make condition-finding efficient.

We hope this series serves as a foothold for you to advance from the stage of "using" additive manufacturing to the stage of "understanding, designing, and optimizing" it.

[← Back to Chapter 4](<./chapter-4.html>) [Series Index →](<./index.html>)

## References

  1. Gibson, I., Rosen, D., & Stucker, B. (2021). _Additive Manufacturing Technologies_ (3rd ed.). Springer. - Standard textbook covering AM data processing, slicing, and simulation
  2. ISO/ASTM 52900:2021. _Additive manufacturing — General principles — Fundamentals and vocabulary_. - International standard for AM process classification and terminology
  3. Incropera, F.P., DeWitt, D.P., Bergman, T.L., & Lavine, A.S. (2017). _Fundamentals of Heat and Mass Transfer_ (8th ed.). Wiley. - Standard textbook on transient heat conduction and the explicit method and stability condition
  4. Harris, C.R., et al. (2020). "Array Programming with NumPy." _Nature_ , 585, 357-362. - Foundational paper on NumPy, which underpins the mesh and numerical computation in this chapter
  5. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_ , 12, 2825-2830. - Foundational reference for scikit-learn used in Code Example 5
  6. Dawson-Haggerty, M., et al. (2024). _trimesh: Python library for loading and using triangular meshes_. <https://trimesh.org/> \- Comprehensive library for real STL processing, mesh repair, and visualization
  7. Slic3r Project. (2024). _Slic3r Manual: Slicing Algorithms_. <https://manual.slic3r.org/> \- Implementation notes on plane-intersection slicing and contour generation

## Tools and Libraries Used

  * **NumPy** (v1.24+): Numerical computing library (mesh analysis, slicing, heat conduction) - <https://numpy.org/>
  * **scikit-learn** (v1.3+): Machine learning library (random forest regression) - <https://scikit-learn.org/>
  * **trimesh** (v4.0+): 3D mesh processing library (for further study) - <https://trimesh.org/>
  * **Python** (v3.10+): Runtime for the code examples in this chapter - <https://www.python.org/>

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
