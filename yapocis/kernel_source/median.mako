%for npoints in steps:
    float median${npoints}(float *a, int n) {
    float l,h;
    bool lt;
    % for i in range(npoints//2+1):
        % for j in range(i+1, npoints):
            lt = a[${j}] < a[${i}];
            l = lt ? a[${j}] : a[${i}];
            h = lt ? a[${i}] : a[${j}];
            a[${i}] = l;
            a[${j}] = h;
        %endfor
    %endfor
    return a[n/2];
    }
%endfor

float median(float *a, int n) {
%for npoints in steps:
    if(n <= ${npoints}) return median${npoints}(a, n);
%endfor
return(0.0);
}

