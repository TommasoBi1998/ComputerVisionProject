variable1 = video(1).labelled_frames(10).myleft;
variable2 = video(1).labelled_frames(10).myright;
variable3 = video(1).labelled_frames(10).yourleft;
variable4 = video(1).labelled_frames(10).yourright;
fileName = "mask_0206.yml";

[rows1, cols1] = size(variable1);
[rows2, cols2] = size(variable2);
[rows3, cols3] = size(variable3);
[rows4, cols4] = size(variable4);
cols = 2;

variable = [variable1; [-1,-1]; variable2; [-1,-1]; variable3; [-1,-1]; variable4];

% Beware of Matlab's linear indexing
variable = variable';
[~, rows] = size(variable);

% Write mode as default
if ( ~exist('flag','var') )
    flag = 'w'; 
end

if ( ~exist(fileName,'file') || flag == 'w' )
    % New file or write mode specified 
    file = fopen( fileName, 'w');
    fprintf( file, '%%YAML:1.0\n');
else
    % Append mode
    file = fopen( fileName, 'a');
end

% Write variable header
fprintf( file, '    %s: !!opencv-matrix\n', "mask");
fprintf( file, '        rows: %d\n', rows);
fprintf( file, '        cols: %d\n', cols);
fprintf( file, '        dt: f\n');
fprintf( file, '        data: [ ');

% Write variable data
for i=1:rows*2
    fprintf( file, '%.6f', variable(i));
    if (i == rows*2), break, end
    fprintf( file, ', ');
    if mod(i+1,4) == 0
        fprintf( file, '\n            ');
    end
end

fprintf( file, ']\n');
fclose(file);