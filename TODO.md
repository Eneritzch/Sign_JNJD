# TODO: Rediseño de static.html

## Información Recopilada
- Archivo actual: templates/recognition/static.html
- Paleta de colores: --color-primary (#baa4cb), --color-primary-dark (#a99cc3), etc.
- Layout actual: Grid de 3 columnas (principal y lateral)
- Funcionalidad: Reconocimiento de señas con video, captura, detección de mano, resultados
- Lógica JS: Intacta, maneja video, captura, reconocimiento

## Plan de Rediseño
- Reestructurar a layout de columna principal con secciones apiladas
- Panel de cámara más prominente (full width)
- Resultados en overlay modal
- Stats organizados en grid inferior
- Mantener paleta de colores y estilos existentes
- No afectar otros HTML

## Pasos a Completar
- [x] Reestructurar HTML: Cambiar grid a columna principal
- [x] Hacer panel de cámara full width y prominente
- [x] Crear modal overlay para resultados
- [x] Organizar stats en grid inferior
- [x] Ajustar clases CSS para nuevo layout
- [x] Verificar que lógica JS funcione (mostrar/ocultar elementos)
- [x] Probar diseño en diferentes tamaños de pantalla
- [x] Actualizar JavaScript para usar modal en lugar del panel lateral
- [x] Agregar event listeners para cerrar modal
