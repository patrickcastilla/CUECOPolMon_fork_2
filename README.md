# Política Monetaria y Aplicaciones
Este repositorio ha sido creado para compilar todos los códigos desarrollados como apoyo para el curso de 'Política Monetaria y Aplicaciones', que forma parte del curso de verano en economía del Banco Central de Reserva del Perú. Este repositorio sólo estará disponible hasta el final del curso.
El repositorio está organizado de la siguiente manera:
1. La carpeta '01_NotasDeClase' contiene las versiones actualizadas de las notas de clase.
2. La carpeta '02_Libraries' contiene todos las librerias --archivos de extensión '.py'-- dessarrolladas para este curso y que se utilizan en los notebooks Jupyter.
   * DSGE_PreProcessing.py: Funciones que leen el archivo de extensión '.txt' con información de la estructura del modelo. Las rutinas en esta librería utilizan SimPy para identificar al modelo y reescribirlo en el formato,
   $f(\mathbf{y',y,y^{\ell}},e)=0$ con $e\sim(0,\Sigma)$. Posteriormente, con las primeras derivadas se computa la aproximación de primer orden, $f_{0}+f_{y'}y'+f_{y}y+f_{y^{\ell}}y^{\ell}+f_ee=0$. Note que estos son objetos simbólicos derivados con SimPy. Finalmente, los objetos simbólicos, $f_{0}$, $f_{y'}$, $f_{y}$, $f_{y^{\ell}}$ y $f_e$, se convierten en funciones de python (i.e., 'def').
   * DSGE_LinearMethods.py: Con las funciones creadas 'DSGE_PreProcessing.py' puede hacer los siguiente:
     * Resolver el modelo: Esto implica, cargar la calibración (otro archivo '.txt') para obtener los valores puntuales que toman $f_{0}$, $f_{y'}$, $f_{y}$, $f_{y^{\ell}}$ y $f_e$ y computar la solución numérica: $y = h_0 + h_yy^{\ell} + h_ee$
     * Calcular: FIRs, filtro de kalman y hacer proyecciones.
   * HandleTimeSeries.py: Permite manipular con facilidad las series de tiempo con Pandas. 
3. La carpeta '03_DB' contiene los archivos excel con los datos utilizados como ejemplo.
4. La carpeta '04_Modelo' los archivos ".txt" con la información del modelo y de la calibración de coeficientes
4. El notebook principal es '01_Clase_sobre_MPT.ipynb'
