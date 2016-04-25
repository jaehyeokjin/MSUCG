% Initialization
r6=[0.0,2.4,5.0;0.0,2.5,5.0;0.0,2.5,5.0];
r=[0.0,5.0,7.0;0.0,0.0,0.0;0.0,0.0,0.0];
r3=[0.0,5.0,10.0;0.0,0.0,0.0;0.0,0.0,0.0];
r13=[0.0,2.0,4.0;0.0,0.0,0.0;0.0,0.0,0.0];
distance=zeros(3,3);
w=zeros(1,3);
p=zeros(2,3);
S=zeros(3,3);
    % Subforce term
f_1=zeros(3,3);
f_2=zeros(3,3);
f_31=zeros(3,3);
f_33=zeros(3,3);

lj=6.0;
c=4.0; % sigma cutoff
pc=1.0; % p constant
    %LJ coefficient
eps=[0.48, 0.36];
sig=[4.0, 4.4];
epsilon=zeros(2,2);
sigma=zeros(2,2);

% Distance calculation
for i=1:3
    for j=1:3
        if (i ~= j)
            distance(i,j) = sqrt((r(1,i)-r(1,j))^2.0+(r(2,i)-r(2,j))^2.0+(r(3,i)-r(3,j))^2.0);
        end
    end
end

% LJ force/potential calculation
for a=1:2
    for b=1:2
        epsilon(a,b) = sqrt(eps(a)*eps(b));
        sigma(a,b) = (sig(a)+sig(b))/2.0;
    end
end

% W value calculation
for i=1:3
    for j=1:3
        if (i~=j)
            w(i) = w(i) + 0.5* (1-tanh((distance(i,j)-c)/(0.1 * c)));
        end
    end
end

% P value calculation
for i=1:3
    for j=1:2
        p(j,i) = 0.5 + 0.5*tanh((w(i)-pc)/(0.1*pc)) * (-1)^(j+1);
    end
end

% Derivative of P Calculation
for i=1:3
    for j=1:3
        if (i~=j)
            for k=1:3
                S(k,i) = S(k,i) + (sech((distance(i,j)-c)/(0.1*c)))^2.0 * (r(k,i) - r(k,j))/distance(i,j);
            end
        end
    end
end

% First subforce calculation
for i=1:3
    for j=1:3
        if(i~=j)
            if(distance(i,j) < lj)
                for a=1:2
                    for b=1:2
                        lj_force = 48.0 * epsilon(a,b) * (sigma(a,b)^12)/(distance(i,j)^12) - 24.0 * epsilon(a,b) * (sigma(a,b)^6.0)/(distance(i,j)^6.0);
                        for k=1:3
                            f_1(k,i) = f_1(k,i) +  p(a,i) * p(b,j) * lj_force/(distance(i,j)^2) * (r(k,i)-r(k,j));
                        end
                    end
                end
            end
        end
    end
end
f_temp = 0.0;

% Second subforce calculation
for i=1:3
    for j=1:3
        if(i~=j)
            if(distance(i,j) < lj)
                for a=1:2
                    for b=1:2
                        lj_potential = 4 * epsilon(a,b) * (sigma(a,b)^12)/(distance(i,j)^12)- 4*epsilon(a,b) * (sigma(a,b)^6)/(distance(i,j)^6);
                        for k=1:3
                            s_head_i = (sech((w(i)-pc)/(0.1*pc)))^2.0 / (0.04*c*pc);
                            s_head_j_1 = (sech((w(j)-pc)/(0.1*pc)))^2.0 / (0.04*c*pc);
                            f_2(k,i) = f_2(k,i) + (s_head_i * S(k,i) * (-1)^(a) * p(b,j) + p(a,i) * s_head_j_1 * (-1)^(b) * (sech((distance(i,j)-c)/(0.1*c)))^2.0 * (r(k,i)-r(k,j))/distance(i,j)) * lj_potential;
                        end
                    end
                end
            end
        end
    end
end

% third subforce calculation
% third_1
for i=1:3
    for j=1:3
        if (i~=j)
            if(distance(i,j) < lj)
                for k=1:3
                    if (j~=k) && (i~=k)
                        if(distance(j,k) < lj)
                            if (distance(i,k) < lj)
                                for a=1:2
                                    for b=1:2
                                        lj_potential = 4 * epsilon(a,b) * (sigma(a,b)^12)/(distance(j,k)^12)- 4*epsilon(a,b) * (sigma(a,b)^6)/(distance(j,k)^6);
                                        for l=1:3
                                            s_head_j = (sech((w(j)-pc)/(0.1*pc)))^2.0 / (0.04*c*pc);
                                            s_head_k = (sech((w(k)-pc)/(0.1*pc)))^2.0 / (0.04*c*pc);
                                            a_part = p(a,j) * s_head_k * (-1)^(b) * (sech((distance(i,k)-c)/(0.1*c)))^2.0 * (r(l,i)-r(l,k))/distance(i,k);
                                            b_part = p(b,k) * s_head_j * (-1)^(a) * (sech((distance(i,j)-c)/(0.1*c)))^2.0 * (r(l,i)-r(l,j))/distance(i,j);
                                            f_31(l,i) = f_31(l,i) + (a_part+b_part)*0.5* lj_potential;
                                        end
                                    end
                                end
                            else
                                for a=1:2
                                    for b=1:2
                                        lj_potential = 4 * epsilon(a,b) * (sigma(a,b)^12)/(distance(j,k)^12)- 4*epsilon(a,b) * (sigma(a,b)^6)/(distance(j,k)^6);
                                        for l=1:3
                                            s_head_jj = (sech((w(j)-pc)/(0.1*pc)))^2.0 / (0.04*c*pc);
                                            bb_part = s_head_jj * (sech((distance(i,j)-c)/(0.1*c)))^2.0 * (r(l,i)-r(l,j))/distance(i,j) * (-1)^(a) * p(b,k);
                                            f_33(l,i) = f_33(l,i) + (bb_part)* lj_potential;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

f_3 = f_31+f_33;
f_total = f_1 + f_2 + f_3
% third_2