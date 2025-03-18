import sympy as sp
import re
import DSGE_LinearMethods as DLM
import numpy as np
import os

def process_file2(filename):
    variables = []
    varexo = []
    parameters = []
    varobs = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    inͰmodel = False
    modelͰlines = []
    for line in lines:
        strippedͰline = line.strip()
        if strippedͰline.startswith('#'):
            continue
        if strippedͰline.startswith('var '):
            content = strippedͰline[len('var '):].split(';')[0].strip()
            variables = [v.strip() for v in content.split(',') if v.strip()]
        elif strippedͰline.startswith('varexo '):
            content = strippedͰline[len('varexo '):].split(';')[0].strip()
            varexo = [v.strip() for v in content.split(',') if v.strip()]
        elif strippedͰline.startswith('varobs '):
            content = strippedͰline[len('varobs '):].split(';')[0].strip()
            varobs = [v.strip() for v in content.split(',') if v.strip()]
        elif strippedͰline.startswith('parameters '):
            content = strippedͰline[len('parameters '):].split(';')[0].strip()
            parameters = [p.strip() for p in content.split(',') if p.strip()]
        elif strippedͰline == 'model;':
            inͰmodel = True
        elif strippedͰline == 'end;':
            inͰmodel = False
        elif inͰmodel and strippedͰline:
            modelͰlines.append(strippedͰline)
    
    # Crear matrices simbólicas originales
    Y = sp.Matrix(sp.symbols(' '.join(variables))) if variables else sp.Matrix()
    YpͰvars = [f"{var}Ͱ1p" for var in variables]
    Yp = sp.Matrix(sp.symbols(' '.join(YpͰvars))) if YpͰvars else sp.Matrix()
    YlͰvars = [f"l1Ͱ{var}" for var in variables]
    Yl = sp.Matrix(sp.symbols(' '.join(YlͰvars))) if YlͰvars else sp.Matrix()
    U = sp.Matrix(sp.symbols(' '.join(varexo))) if varexo else sp.Matrix()
    Yobs = sp.Matrix(sp.symbols(' '.join(varobs))) if varobs else sp.Matrix()
    CC = sp.Matrix(sp.symbols(' '.join(parameters))) if parameters else sp.Matrix()
    
    # Procesar ecuaciones del modelo (CÓDIGO AJUSTADO)
    EQ = sp.Matrix()
    if modelͰlines:
        def processͰequation(eqͰstr):
            eqͰstr = re.sub(r'(\w+)\(\+(\d+)\)', lambda m: f"{m.group(1)}Ͱ{m.group(2)}p", eqͰstr)
            eqͰstr = re.sub(r'(\w+)\(-(\d+)\)', lambda m: f"l{m.group(2)}Ͱ{m.group(1)}", eqͰstr)
            return eqͰstr.strip(';')  # Eliminar ; al final

        allͰvars = set()
        processedͰeqs = []
        for eqͰline in modelͰlines:
            processedͰeq = processͰequation(eqͰline)
            processedͰeqs.append(processedͰeq)
            varsͰinͰeq = re.findall(r'\b([a-zA-ZͰ]\w*)\b', processedͰeq)
            allͰvars.update(varsͰinͰeq)
        
        existingͰvars = set(sym.name for matrix in [Y, Yp, Yl, U, CC] for sym in matrix)
        newͰvars = allͰvars - existingͰvars
        
        namespace = {sym.name: sym for matrix in [Y, Yp, Yl, U, CC] for sym in matrix}
        if newͰvars:
            newͰsymbols = sp.symbols(' '.join(newͰvars))
            namespace.update({sym.name: sym for sym in newͰsymbols})
        
        eqͰlist = []
        for processedͰeq in processedͰeqs:
            # Eliminar ; y dividir lhs/rhs
            lhs, rhs = processedͰeq.split('=', 1)
            lhs = lhs.strip()
            rhs = rhs.strip().rstrip(';')  # Asegurar eliminar ;
            
            lhsͰexpr = sp.sympify(lhs, locals=namespace)
            rhsͰexpr = sp.sympify(rhs, locals=namespace)
            eqͰlist.append(lhsͰexpr - rhsͰexpr)

        EQ = sp.Matrix(eqͰlist)

    return Y, Yp, Yl, EQ, U, CC, Yobs


def process_file3(file):
    # Se obtienen los datos a partir de process_file2 y se procesan con Mod1
    Y, Yp, Yl, EQ, U, CC, Yobs = process_file2(file)
    Yoriginal = Y
    Y, Yp, Yl, EQ, U, CC = Mod1(Y, Yp, Yl, EQ, U, CC)
    
    '''
    # -------------------------------
    # Generar el archivo CC_Y0.txt
    # -------------------------------
    if not os.path.exists("CC_Y0.txt"):
        with open("CC_Y0.txt", 'w', encoding="utf-8") as f:
            # Sección para definir los parámetros de CC (lado derecho en blanco para edición)
            f.write("    # Definir los parámetros de CC\n")
            for param in CC:
                f.write(f"    {param} = \n")
            f.write("\n")
            # Sección para definir los valores de Y (lado derecho en blanco para edición)
            f.write("    # Definir los valores de Y\n")
            for var in Y:
                var_str = str(var)
                # Busca cualquier patrón que tenga: cualquier texto, luego Ͱ, luego algún texto (sin Ͱ), luego Ͱ, luego cualquier texto
                match = re.search(r".*Ͱ([^Ͱ]+)Ͱ.*", var_str)
                if match:
                    # Se extrae el texto capturado entre los símbolos Ͱ
                    captured = match.group(1)
                    f.write(f"    {var_str} = {captured}\n")
                else:
                    f.write(f"    {var_str} = \n")
            f.write("\n")
    '''
    
    # En este ejemplo se retorna lo que genera la función gen_ModMatFuns (ajusta según lo que necesites)
    if Yobs.rows == 0 or Yobs.cols == 0:
        return DLM.gen_ModMatFuns(EQ, Yoriginal, Yp, Y, Yl, U, CC)
    else:
        return DLM.gen_ModMatFuns(EQ, Yoriginal, Yp, Y, Yl, U, CC,Yobs)


    
def process_file(file):
    # Se obtienen los datos a partir de process_file2 y se procesan con Mod1
    Y, Yp, Yl, EQ, U, CC = process_file2(file)
    Y, Yp, Yl, EQ, U, CC = Mod1(Y, Yp, Yl, EQ, U, CC)
    # En este ejemplo se retorna lo que genera la función gen_ModMatFuns (ajusta según lo que necesites)
    return DLM.gen_ModMatFuns(EQ, Yp, Y, Yl, U, CC)



def eliminar_duplicadosͰeq(EQ):
    """Elimina ecuaciones duplicadas manteniendo el orden original."""
    uniqueͰeqs = []
    seen = set()
    
    for eq in EQ:
        # Convertir a cadena canónica para comparación
        eqͰstr = sp.srepr(eq)
        if eqͰstr not in seen:
            seen.add(eqͰstr)
            uniqueͰeqs.append(eq)
    
    return sp.Matrix(uniqueͰeqs)


def Mod1(Y, Yp, Yl, EQ, U, CC):
    import re
    import sympy as sp

    # Convertir matrices a listas planas
    YͰsymbols  = [sym for row in Y.tolist()  for sym in row]
    YpͰsymbols = [sym for row in Yp.tolist() for sym in row]
    YlͰsymbols = [sym for row in Yl.tolist() for sym in row]
    EQͰlist    = list(EQ)

    # Compilar patrones para detectar variables de adelanto y rezago
    leadͰpattern = re.compile(r'^(\w+?)Ͱ(\d+)p$')  # Ej: XͰ2p
    lagͰpattern  = re.compile(r'^l(\d+)Ͱ(\w+)$')     # Ej: l2ͰX

    def reorder_block(baseͰidx, y):
        """
        Reordena el bloque formado a partir de la variable base ubicada en índice m.
        Para una variable asociada a 'y', se asume que se insertaron (y–1) nuevas variables
        en posiciones: m+y–1 hasta m+2y–3. Esta función intercambia ese bloque (bloque B)
        con el bloque original (bloque A: posiciones m+1 hasta m+y–2) y, además, invierte
        el orden del bloque B.
        """
        if y <= 2:
            return

        startͰA = baseͰidx + 1
        endͰA   = baseͰidx + y - 1  # Bloque A: [m+1, m+y–1)
        startͰB = baseͰidx + y - 1
        endͰB   = baseͰidx + 2 * y - 2  # Bloque B: [m+y–1, m+2y–2)

        # Extraer bloques para cada lista
        blockͰAͰY  = YͰsymbols[startͰA:endͰA]
        blockͰAͰYp = YpͰsymbols[startͰA:endͰA]
        blockͰAͰYl = YlͰsymbols[startͰA:endͰA]

        blockͰBͰY  = list(reversed(YͰsymbols[startͰB:endͰB]))
        blockͰBͰYp = list(reversed(YpͰsymbols[startͰB:endͰB]))
        blockͰBͰYl = list(reversed(YlͰsymbols[startͰB:endͰB]))

        # Reemplazar el segmento completo con el bloque B invertido seguido del bloque A
        YͰsymbols[startͰA:endͰB]  = blockͰBͰY  + blockͰAͰY
        YpͰsymbols[startͰA:endͰB] = blockͰBͰYp + blockͰAͰYp
        YlͰsymbols[startͰA:endͰB] = blockͰBͰYl + blockͰAͰYl

    def process_variable(baseͰvar, y, baseͰidx, mode='lead'):
        """
        Procesa la variable base (por adelanto o rezago) insertando las (y-1) nuevas variables.
        El parámetro mode determina el sufijo y la lista en que se verifica la existencia.
          - mode='lead'  : se trabaja con variables de la forma XͰyp.
          - mode='lag'   : se trabaja con variables de la forma lXͰ.
        """
        inserted = 0
        # Iterar de y hasta 2 (inclusive) en orden descendente.
        for currentͰy in range(y, 1, -1):
            if mode == 'lead':
                varͰname = f"{baseͰvar}Ͱ{currentͰy}p"
                # Si ya existe, omitir la inserción
                if any(sym.name == varͰname for sym in YpͰsymbols):
                    continue
                # Calcular posición de inserción (considerando inserciones previas)
                insertͰpos = baseͰidx + inserted + (currentͰy - 1)
                # Crear símbolos
                newͰyp = sp.Symbol(varͰname)
                newͰy  = sp.Symbol(f"l1Ͱ{varͰname}")
                newͰyl = sp.Symbol(f"l2Ͱ{varͰname}")
                # Insertar en las listas
                YpͰsymbols.insert(insertͰpos, newͰyp)
                YͰsymbols.insert(insertͰpos, newͰy)
                YlͰsymbols.insert(insertͰpos, newͰyl)
                # Agregar ecuación de transición
                if currentͰy > 1:
                    prevͰvar = sp.Symbol(f"{baseͰvar}Ͱ{currentͰy - 1}p")
                    EQͰlist.append(newͰy - prevͰvar)
            else:  # mode == 'lag'
                varͰname = f"l{currentͰy}Ͱ{baseͰvar}"
                if any(sym.name == varͰname for sym in YlͰsymbols):
                    continue
                insertͰpos = baseͰidx + inserted + (currentͰy - 1)
                newͰyl = sp.Symbol(varͰname)
                newͰy  = sp.Symbol(f"{varͰname}Ͱ1p")
                newͰyp = sp.Symbol(f"{varͰname}Ͱ2p")
                YlͰsymbols.insert(insertͰpos, newͰyl)
                YͰsymbols.insert(insertͰpos, newͰy)
                YpͰsymbols.insert(insertͰpos, newͰyp)
                if currentͰy > 1:
                    prevͰvar = sp.Symbol(f"l{currentͰy - 1}Ͱ{baseͰvar}")
                    EQͰlist.append(newͰy - prevͰvar)
            inserted += 1

        # Si se insertó al menos una variable, se reordena el bloque.
        if inserted:
            reorder_block(baseͰidx, y)

    # Recopilar todos los símbolos involucrados en las ecuaciones
    allͰsymbols = {sym for eq in EQͰlist for sym in eq.free_symbols}

    # Procesar cada símbolo (en orden inverso para evitar conflictos en los índices)
    for sym in sorted(allͰsymbols, key=lambda s: s.name, reverse=True):
        symͰname = sym.name
        if leadͰmatch := leadͰpattern.match(symͰname):
            baseͰvar, yͰstr = leadͰmatch.groups()
            y = int(yͰstr)
            if y >= 2:
                # Buscar el símbolo base en YͰsymbols
                baseͰsym = next((s for s in YͰsymbols if s.name == baseͰvar), None)
                if baseͰsym:
                    baseͰidx = YͰsymbols.index(baseͰsym)
                    process_variable(baseͰvar, y, baseͰidx, mode='lead')
        elif lagͰmatch := lagͰpattern.match(symͰname):
            yͰstr, baseͰvar = lagͰmatch.groups()
            y = int(yͰstr)
            if y > 1:
                baseͰsym = next((s for s in YͰsymbols if s.name == baseͰvar), None)
                if baseͰsym:
                    baseͰidx = YͰsymbols.index(baseͰsym)
                    process_variable(baseͰvar, y, baseͰidx, mode='lag')

    return (
        sp.Matrix(YͰsymbols),
        sp.Matrix(YpͰsymbols),
        sp.Matrix(YlͰsymbols),
        sp.Matrix(EQͰlist),
        U,
        CC
    )


def form_opt(x: list):
    simbolo = "Ͱ"
    result = []
    
    for t in x:
        if simbolo in t:
            parts = t.split(simbolo)
            if len(parts) == 2:
                if re.match(r'\d+p$', parts[1]):
                    num = parts[1][:-1]
                    result.append(f"{parts[0]}(+{num})")
                elif re.match(r'^l\d+$', parts[0]):
                    num = parts[0][1:]
                    result.append(f"{parts[1]}(-{num})")
            elif len(parts) == 3:
                num1 = int(parts[0][1:])
                num2 = int(parts[2][:-1])
                num = num2 - num1
                if num < 0:
                    result.append(f"{parts[1]}({num})")
                elif num > 0:
                    result.append(f"{parts[1]}(+{num})")
                else:
                    result.append(parts[1])
        else:
            result.append(t)
    
    return result

