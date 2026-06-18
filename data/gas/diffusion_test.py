import numpy as np

coarse_maze = np.array([
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,1,1,1,1,1],
    [1,0,0,0,1,1,0,0,0,1],
    [1,0,1,1,1,1,0,1,0,1],
    [1,0,0,1,0,0,0,1,0,1],
    [1,1,0,1,0,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1],
])

building = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

S = 5  # room size
W = 1  # wall thickness

# Define the coarse structure as room/wall slots along each axis.
# Row types for building (10 rows):
# 0=wall, 1=room, 2=room, 3=room, 4=wall, 5=room, 6=room, 7=room, 8=wall, 9=wall
row_sizes = [W, S, S, S, W, S, S, S, W, W]
col_sizes = [W, S, S, S, W, S, S, S, S, W]

H_fine = sum(row_sizes)
W_fine = sum(col_sizes)
maze = np.ones((H_fine, W_fine), dtype=int)

row_offsets = np.cumsum([0] + row_sizes)
col_offsets = np.cumsum([0] + col_sizes)

for r in range(building.shape[0]):
    for c in range(building.shape[1]):
        if building[r, c] == 0:
            maze[row_offsets[r]:row_offsets[r+1],
                 col_offsets[c]:col_offsets[c+1]] = 0

def diffusion_with_windows(
    maze,
    source_pos,
    sink_pos,
    source_strength=1.0,
    D=1.0,
    dt=0.01,
    steps=5000
):
    '''heat equation'''
    h, w = maze.shape
    rho = np.zeros((h, w))

    source_y, source_x = source_pos
    sink_y, sink_x = sink_pos

    for step in range(steps):
        new_rho = rho.copy()

        for y in range(h):
            for x in range(w):

                if maze[y, x] == 1:
                    continue  # wall

                # if maze[y, x] == 2:
                new_rho[sink_y, sink_x] = 0.0  # window

                lap = 0.0
                neighbors = 0

                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if maze[ny, nx] == 1:
                            lap += rho[y, x]  # next to wall
                        else:
                            lap += rho[ny, nx]
                        neighbors += 1

                lap = (lap - neighbors * rho[y, x])

                source = source_strength if (y, x) == (source_y, source_x) else 0.0

                new_rho[y, x] = rho[y, x] + dt * (D * lap + source)

        rho = new_rho

    return rho

factor = 3
# maze = np.kron(building, np.ones((factor,factor)))
# maze = maze.astype(int)
# h, w = maze.shape


D = 1.0                   # diffusion coefficient
dx = 1.0 / (maze.shape[0]/building.shape[0])  # physical spacing
dt = 0.1
steps = 50000
source_pos = (6*factor, 2*factor)  # same physical source position
sink_pos = (2 * factor, 8 * factor)
source_strength = 5.0

density = diffusion_with_windows(maze,
                                 source_pos=source_pos,
                                 sink_pos=sink_pos,
                                 D=D,
                                 dt=dt,
                                 steps=steps,
                                 source_strength=source_strength
                                 )
print(density.shape)
import matplotlib.pyplot as plt

plt.imshow(density, origin="upper")
plt.colorbar(label="Gas density")
plt.title("Gas diffusion with source and sink")
plt.scatter(2*factor, 6*factor, c="red", label="Source")
plt.scatter(8*factor,2* factor,c='k', label='Sink')
plt.legend()
# plt.show()
plt.savefig('figs/gas_building_diffusion.png', dpi=300, bbox_inches='tight')

np.savetxt('data/gas_building_diffusion.csv', density, delimiter=',')
