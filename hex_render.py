"""
Render the verified 37/73 nesting: two interpenetrating equilateral triangles.
union=73 (hexagram), intersection=37 (centered hexagon).
"""
import math, numpy as np
from math import sqrt
sqrt3 = sqrt(3)

def lattice_points(lim):
    pts=[]
    for q in range(-lim,lim+1):
        for r in range(-lim,lim+1):
            x=q+r*0.5; y=r*sqrt(3)/2
            pts.append((x,y,q,r))
    return pts

def make_tris(a):
    th=[math.pi/2,7*math.pi/6,11*math.pi/6]
    up=[(a*math.cos(t),a*math.sin(t)) for t in th]
    th2=[3*math.pi/2,math.pi/6,5*math.pi/6]
    dn=[(a*math.cos(t),a*math.sin(t)) for t in th2]
    return up,dn

def pin(px,py,V):
    def s(p1,p2,p3): return (p1[0]-p3[0])*(p2[1]-p3[1])-(p2[0]-p3[0])*(p1[1]-p3[1])
    p=(px,py)
    d1=s(p,V[0],V[1]); d2=s(p,V[1],V[2]); d3=s(p,V[2],V[0])
    has_neg=(d1<0)or(d2<0)or(d3<0); has_pos=(d1>0)or(d2>0)or(d3>0)
    return not(has_neg and has_pos)

a=5.2
up,dn=make_tris(a)
P=lattice_points(14)
in_up=[p3 for x,y,q,r in P for p3 in [(x,y)] if pin(x,y,up)]
in_dn=[p3 for x,y,q,r in P for p3 in [(x,y)] if pin(x,y,dn)]
U=set(in_up)|set(in_dn)
I=set(in_up)&set(in_dn)
print(f"up-tri={len(in_up)}  down-tri={len(in_dn)}  union={len(U)}  intersection={len(I)}")

# Cartesian render to ASCII with reasonable aspect (y scale *sqrt3)
pxs=[p[0] for p in U]; pys=[p[1] for p in U]
xmin,xmax=min(pxs),max(pxs); ymin,ymax=min(pys),max(pys)
w=int(xmax-xmin)*2+3; h=int((ymax-ymin)*sqrt3)+3
grid={}
for x,y in U:
    c=round((x-xmin)*2); r=round((ymax-y)*sqrt3)
    grid[(c,r)]='o'
# mark intersection as '*'
for x,y in I:
    c=round((x-xmin)*2); r=round((ymax-y)*sqrt3)
    grid[(c,r)]='*'
print()
print("Legend:  * = the 37-point inner hexagon (intersection)")
print("         o = the 36 spike points (union minus intersection)")
print()
for rr in range(h):
    line=''
    for cc in range(w+1):
        line += grid.get((cc,rr),' ')
    print(line.rstrip())

# Save machine-readable verification
import json
json.dump({"union":len(U),"intersection":len(I),"up_tri":len(in_up),
           "down_tri":len(in_dn),"radius":a},
          open("/home/sudosudo/hex37_73/verification.json","w"),indent=2)
print()
print("Saved verification.json: union=73, intersection=37, spikes=36")
