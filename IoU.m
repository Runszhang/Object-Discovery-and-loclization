function o = IoU(a, b)
% a  a single bounding box
% b  a single bounding box

a(3) = a(3) + a(1);
a(4) = a(4) + a(2);
b(3) = b(3) + b(1);
b(4) = b(4) + b(2);

x1 = max(a(1), b(1));
y1 = max(a(2), b(2));
x2 = min(a(3), b(3));
y2 = min(a(4), b(4));

w = max(x2 - x1, 0);
h = max(y2 - y1, 0);
inter = w * h;
aarea = (a(3) - a(1)) * (a(4) - a(2));
barea = (b(3) - b(1)) * (b(4) - b(2));
o = inter ./ (aarea+barea-inter);

