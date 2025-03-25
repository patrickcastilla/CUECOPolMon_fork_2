# Política Monetaria y Aplicaciones
Este repositorio ha sido creado para compilar todos los códigos desarrollados como apoyo para el curso de 'Política Monetaria y Aplicaciones', que forma parte del curso de verano en economía del Banco Central de Reserva del Perú. Este repositorio sólo estará disponible hasta el final del curso.
El repositorio está organizado de la siguiente manera:
1. La carpeta '01_NotasDeClase' contiene las versiones actualizadas de las notas de clase.
2. La carpeta '02_Libraries' contiene todos las librerias --archivos de extensión '.py'-- dessarrolladas para este curso y que se utilizan en los notebooks Jupyter.
   * DSGE_PreProcessing.py: Funciones que leen el archivo de extensión '.txt' con información de la estructura del modelo. Las rutinas en esta librería utilizan SimPy para identificar al modelo y reescribirlo en el formato,
   $f(\mathbf{y',y,y^{\ell}},e)=0$ con $e\sim(0,\Sigma)$. Posteriormente, con las primeras derivadas se computa la aproximación de primer orden, $f_{y'}y'+f_{y}y+f_{y^{\ell}}y^{\ell}+f_ee=0$. Note que estos son objetos simbólicos derivados con SimPy. Finalmente, los objetos simbolicos, $f_{y'}$,$y'$,$f_{y}$,$y$,$f_{y^{\ell}}$,$y^{\ell}$,$f_e$ y $e$, se convierten en funciones de python (i.e., 'def').
   * DSGE_LinearMethods.py:
   * HandleTimeSeries.py:
3. La carpeta '03_DB' contiene los archivos excel con los datos utilizados como ejemplo.
4. La carpeta '04_Modelo' los archivos ".txt" con la información del modelo y de la calibración de coeficientes
4. El notebook principal es '01_Clase_sobre_MPT.ipynb'
