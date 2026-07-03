<p align="center">
    <a href="https://www.latex-project.org"><img src="https://img.shields.io/badge/Made%20with-LaTeX-red.svg?style=flat-square"></a>
    <a href="https://www.latex-project.org/lppl/lppl-1-3c"><img src="https://img.shields.io/badge/License-LPPL%20v1.3c-yellow.svg?style=flat-square"></a>
    <a href="https://github.com/asalber/plantilla-tfg-ceu/releases"><img src="https://img.shields.io/github/v/tag/asalber/plantilla-tfg-ceu?style=flat-square&label=Release&color=8892BF"></a>
    <a href="https://github.com/asalber/plantilla-tfg-ceu/graphs/commit-activity"><img src="https://img.shields.io/badge/Maintained-Yes-brightgreen.svg?style=flat-square"></a>
    <a href=""><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=flat-square"></a>
</p>
<br/>

CEUTFG es una plantilla de LaTeX de código abierto **diseñada para crear tesis, trabajos de fin de grado** específicamente adaptada para los estudiantes de la [Universidad CEU San Pablo](https://www.uspceu.com/). Fue desarrollada a partir de la plantilla [IPLeiria Thesis](https://github.com/joseareia/ipleiria-thesis) por lo que hereda la mayoría de sus características. Permite un _diseño limpio, estéticamente agradable y profesional_ mientras se mantiene altamente personalizable para adaptarse a diversas necesidades. La plantilla también es fácil de usar, lo que la hace accesible incluso para principiantes.

Si quieres ver el aspecto final aquí tienes una ver una [DEMOSTRACIÓN](/Assets/TFG.pdf).

![Ejemplo de la portada del TFG](Assets/ejemplo-portada-tfg.png)

Este respositorio contiene todo el _código fuente de la plantilla_, organizado en una jerarquía de archivos clara y bien estructurada. También incluye varias herramientas y _archivos de configuración necesarios para compilar_ la plantilla en diferentes entornos de trabajo.

## Instrucciones de instalación y configuración

Esta plantilla está disponible en línea en [Overleaf](https://www.overleaf.com/latex/templates/polytechnic-university-of-leiria-thesis-template/tqgbrncfhwgt), proporcionando una forma sencilla de compilarla sin necesidad de instalar dependencias. 

Pero también puedes descargarla y usarla localmente. Para ello, asegúrate de tener instaladas las dependencias necesarias y sigue las siguientes instrucciones.

### Dependencias

Para usar esta plantilla localmente, asegúrate de tener instaladas las siguientes dependencias para garantizar una experiencia fluida y sin problemas al utilizar la plantilla:

#### Fuentes

- **Fuente Lato**: Requerida para el estilo del documento.
    - En sistemas basados en Debian, instálala con: `sudo apt install fonts-lato`.
    - Alternativamente, descárgala directamente desde [Google Fonts](https://fonts.google.com/specimen/Lato?query=lato).

- **Fuentes Texgyre**: Esenciales para estilos tipográficos específicos.
    - En sistemas basados en Debian, instálalas con: `sudo apt install fonts-texgyre`.

#### Herramientas

- **Rubber** e **inotify-tools**: Necesarios para la compilación automatizada. Instálalos con: `sudo apt install rubber inotify-tools`

- **Make**: Requerido para ejecutar los scripts de compilación. Instálalo con: `sudo apt install make`

- **Latexmk**: Utilizado para compilación automatizada y parsing de argumentos. Instálalo con: `sudo apt install perl latexmk`

### Compilación

Para compilar el documento hay varias opciones disponibles.

#### Opción 1: Compilar localmente

Para compilar el proyecto primero debes instalar LaTeX en tu sistema. Aquí tienes cómo hacerlo:

- En Linux y Windows, instala [TeX Live](https://www.tug.org/texlive/) o [MikTeX](https://miktex.org/).
- En macOS, instala [MacTeX](https://www.tug.org/mactex/).

Una vez instalado LaTeX, puedes compilar la plantilla usando los siguientes métodos con el `Makefile` proporcionado:

- Para compilar el proyecto con `latexmk` (opción por defecto), ejecuta: `make`.
- Para compilar el proyecto con `rubber`, ejecuta: `make TOOL=rubber`.
- Para limpiar el directorio para ambos herramientas de compilación, ejecuta: `make clean`. Sin embargo, si quieres usar el proceso de limpieza específico de cada herramienta de compilación, también puedes ejecutar: `make TOOL=latexmk clean` o `make TOOL=rubber clean`.

Ten en cuenta que el archivo de configuración `.latexmk` ya está preparado y no debe ser modificado.

Para más detalles sobre la elección de `rubber`, consulta [este issue](https://github.com/joseareia/ipleiria-thesis/issues/3#issuecomment-2463429883).

#### Opción 2: VSCode LaTeX Workshop con DevContainer

Para una experiencia de desarrollo más sencilla, puedes usar la función **DevContainer** de VSCode, que configura un entorno basado en Docker con todas las herramientas LaTeX necesarias preinstaladas. Para empezar:

1. Instala [Docker](https://docs.docker.com/engine/install/) y [VSCode](https://code.visualstudio.com/download).
2. Abre el directorio de este proyecto en VSCode.
3. Instala el paquete de extensiones **Remote Development** desde el panel de extensiones de VSCode.
4. Pulsa `Ctrl+Shift+P` y ejecuta el comando `Dev Container: Open Folder in Container`.
5. Abre `IPLeiriaMain.tex`, haz clic en el icono de TeX en la barra lateral y selecciona `Build LaTeX Project`.

#### Opción 3: GitHub Codespaces

Si tienes acceso a **GitHub Codespaces**, puedes trabajar en el documento LaTeX directamente en el navegador sin instalar nada en tu equipo. Simplemente:

1. Ve al menú desplegable Code del repositorio.
2. Selecciona **Codespaces** y haz clic en el botón `+` para crear un nuevo codespace.


## Instrucciones para Usar la Plantilla

Antes de comenzar a usar esta plantilla, es importante entender su estructura y cómo personalizarla para adaptarla a tus necesidades. A continuación, se describen los pasos esenciales para utilizar esta plantilla de manera efectiva.

### Archivos y directorios

La plantilla consta de varios directorios y archivos, incluyendo siete directorios distintos y numerosos archivos. Los dos archivos más importantes son `IPLeiriaMain.tex` y `IPLeiriaThesis.cls`. A continuación se muestra una tabla que describe los distintos directorios, junto con una descripción de cada uno y un indicador de si debes modificar su contenido. Una marca de verificación significa que el directorio puede modificarse, mientras que un guion indica que no debe cambiarse.

| Directorio | Modificable | Descripción |
|---|:---:|---|
| `Bibliography` | :white_check_mark: | Este directorio contiene el archivo de bibliografía utilizado para gestionar las referencias a lo largo del documento. |
| `Chapters` | :white_check_mark: | Este directorio contiene los capítulos individuales de la tesis, lo que facilita trabajar en ellos por separado. |
| `Code` | :white_check_mark: | Este directorio contiene los ejemplos de código y los scripts relevantes del trabajo.  |
| `Configurations` | - | Este directorio contiene todos los archivos de configuración requeridos para la plantilla, como la disposición y la configuración de estilo. |
| `Img` | :white_check_mark: | Este directorio contiene todas las imágenes referenciadas en el documento. |
| `Matter` | - | Este directorio contiene material preliminar del documento, incluida la portada, la declaración de derechos de autor y el glosario, etc. |
| `Metadata` | :white_check_mark: | Este directorio contiene el archivo de metadatos, donde se pueden personalizar detalles clave del documento, como el autor, el título y el director. |

>[!NOTE]
>Aunque el cuadro anterior indica que el directorio `Matter` no es modificable, dos archivos dentro de ese directorio deben ser alterados cuando sea necesario: `04-Glossary.tex` y `05-Acronyms.tex`. Aunque los nombres son bastante autoexplicativos, estos archivos deben contener las entradas del glosario y de los acrónimos, respectivamente.

Es importante tener en cuenta que los archivos están organizados según una convención de nombres específica, que debe ser _respetada_. La convención de nombres consiste en un valor numérico ascendente de dos dígitos, seguido de un guion y luego el nombre del archivo empezando por mayúscula. El nombre debe intentar ser siempre una sola palabra. Si se necesita más de una palabra, deben separarse con un guion.

>[!WARNING]
>Los dos archivos mencionados anteriormente, `TFG.tex` y `CEUTFG.cls`, deben usarse con precaución. El archivo principal, como su nombre indica, es el archivo maestro donde añadirás los capítulos necesarios para incluir en tu trabajo. El archivo de clase, por otro lado, requiere aún más precaución, y no se recomienda modificarlo salvo que tengas conocimientos avanzados de LaTeX.

### Opciones de la clase de la plantilla

El primer comando del fichero `TFG.tex` es el comando `\texttt{documentclass}`, que carga la clase personalizada para esta plantilla. Esta clase permite personalizar el contenido y el aspecto final del documento a través de distintos parámetros opcionales que se pueden especificar. Las opciones disponibles se enumeran a continuación.

| Opciones | Descripción |
|---|---|
| **language = OPT** <br> `spanish`, `english` | **Selección del idioma preferido.** <br> ⇨ _Por defecto:_ `language=english` |
| **chapterstyle = OPT** <br> `classic`, `modern`, `fancy` | **Selección del estilo de diseño de los capítulos.** <br> ⇨ _Por defecto:_ `chapterstyle = classic` <br> _Esta opción modifica la apariencia del capítulo, incluyendo el estilo del título y numeración._ |
| **coverstyle = OPT** <br> `classic`, `bw` | **Elección de un estilo para la portada.** <br> ⇨ _Por defecto:_ `coverstyle = bw` <br> `bw` ⇨ _Hace la portada en blanco y negro._ <br> `classic` ⇨ _Muestra la portada en el azul._ |
| **docstage = OPT** <br> `final`, `working` | **Elección de la etapa del documento.** <br> ⇨ _Por defecto:_ `docstage = final` <br> `final` ⇨ _Asume que esta es la versión final del documento._ <br> `working` ⇨ _Asume que el documento está en progreso (requiere `\DocumentVersion{}`)._ |
| **media = OPT** <br> `paper`, `screen` | **Tipo de medio del proyecto.** <br> ⇨ _Por defecto:_ `media=paper` <br> `paper` ⇨ _Aparecerán páginas en blanco entre secciones._ <br> `screen` ⇨ _No aparecerán páginas en blanco entre secciones._ |
| **linkcolor = OPT** <br> `color` | **Color principal del tema.** <br> ⇨ _Por defecto:_ `linkcolor = blue` (#003CA3)<br> _Esta opción requiere un nombre de color válido. Consulta el manual de `xcolor` para seleccionar un color válido._ |
| **bookprint = OPT** <br> `true`, `false` | **Para impresión en libro.** <br> ⇨ _Por defecto:_ `bookprint = false` <br> _Esta opción añade un margen de encuadernación en las páginas impares para permitir la impresión, ya que aumenta el margen izquierdo._ |
| **aiacknowledgement = OPT** <br> `true`, `false` | **Impresión de reconocimiento de IA.** <br> ⇨ _Por defecto:_ `aiacknowledgement = true` <br> _Esta opción añade una sección destinada a que el usuario inserte su reconocimiento del uso de IA._ |

### Personalización de metadatos

Después de definir las opciones de la clase del documento, hay que configurar algunos metadatos como el autor, el título, el director, etc. Dado que esta plantilla admite una amplia gama de opciones de metadatos, se proporciona un archivo dedicado para este propósito. El archivo `Metadata/Metadata.tex` enumera las variables de metadatos, con comentarios sobre si son obligatorias. Comenta las variables para omitirlas. A continuación se muestra una tabla que enumera todas las variables de metadatos disponibles, su comando `GET` (utilizado para recuperar información automáticamente) y si son obligatorias para el documento.

| Variable | Comandos de macro | Obligatorio |
|---|---|:---:|
| Título | `\GetTitle` | :white_check_mark: |
| Subtítulo | `\GetSubtitle` | :white_check_mark: |
| Universidad | `\GetUniversity` | :white_check_mark: |
| Escuela | `\GetSchool` | :white_check_mark: |
| Departamento | `\GetDepartment` | :white_check_mark: |
| Título académico | `\GetDegree` | :white_check_mark: |
| Curso | `\GetCourse` | - |
| Local y fecha | `\GetDate` | :white_check_mark: |
| Año académico | `\GetAcademicYear` | :white_check_mark: |
| Nombre autor | `\GetFirstAuthor` | :white_check_mark: |
| Nombre del director | `\GetSupervisor` | :white_check_mark: |
| Correo electrónico del director | `\GetSupervisorMail` | :white_check_mark: |
| Título y afiliación del director | `\GetSupervisorTitle` | :white_check_mark: |
| Nombre del codirector | `\GetCoSupervisor` | - |
| Correo del codirector | `\GetCoSupervisorMail` | - |
| Título y afiliación del codirector | `\GetCoSupervisorTitle` | - |

### Inserción de capítulos

Como ya se ha mencionado, para usar esta plantilla necesitas hacer tres cosas: establecer las opciones apropiadas en la clase del documento, actualizar los metadatos del documento y crear e importar tus capítulos personalizados. Para crear e importar un capítulo personalizado basta con crear un archivo TeX en el directorio `Chapters` que siga la convención de nombres predefinida y después inclúyelo en el archivo principal usando el comando `\include{Capítulo}`.

## Ayuda y Soporte
Si tienes alguna pregunta sobre la plantilla, su uso o si encuentras algún error, no dudes en abrir un [issue](https://github.com/asalber/plantilla-tfg-ceu/issues), iniciar una nueva [discusión](https://github.com/asalber/plantilla-tfg-ceu/discussions) o enviarme un correo electrónico a <a href="mailto:asalber@ceu.es">asalber@ceu.es</a>.

## Contribuciones
¡Las contribuciones a esta plantilla son bienvenidas! Si encuentras algún problema, tienes sugerencias para mejoras o te gustaría agregar nuevas funciones, por favor envía una [pull request](https://github.com/asalber/plantilla-tfg-ceu/pulls). Agradecemos tus comentarios y contribuciones para hacer que esta plantilla sea aún mejor.

## Licencia
El proyecto **TFGCEU** se publica bajo los términos de la [Licencia LPPL 1.3c](https://www.latex-project.org/lppl/lppl-1-3c/).
