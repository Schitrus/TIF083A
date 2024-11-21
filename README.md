## Bevaring av rörelsemängd, energi och rörelsemängdsmoment i stötar
Denna github sida innehåller kod som användes för att behandla insamlad positions data från Qualisys höghastighetskameror.

`parse_script.py` läser .tsv filerna och sparar positionsdatan för alla filer i dictionary format, med arrays av positions data, både för markörer i masscentrum och "rotationsmarkörer".

Därefter är `animate_experiment.py` den huvudsakliga koden som gör alla uträkningar för storheterna: rörelsemängd, energi och rörelsemängdsmoment. `animate_experiment.py` animerar även händelseförloppen för alla mätserier. Här kan de redan genererade animationerna kommas åt i en drivemapp: [https://drive.google.com/drive/folders/18r5qT_Mfxas77JeRF2ToI6Z0qdzpwZks]([https://link-url-here.org](https://drive.google.com/drive/folders/18r5qT_Mfxas77JeRF2ToI6Z0qdzpwZks)).
