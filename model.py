import torch.nn as nn
from torchvision import models

CLASS_NAMES = ['eixample', 'gothic_quarter', 'modernista', 'olympic_modern']

THRESHOLDS = {
    'eixample': 0.571,
    'gothic_quarter': 0.350,
    'modernista': 0.425,
    'olympic_modern': 0.529
}

DESCRIPTIONS = {
    'eixample': "The Eixample district was designed by Ildefons Cerdà in 1860 as a rational grid expansion of Barcelona. Its buildings are characterised by regular rectangular facades, uniform balconies, and chamfered corners at intersections. The style is broadly eclectic and historicist — functional 19th century bourgeois architecture without strong ornamental character.",
    'gothic_quarter': "The Gothic Quarter is Barcelona's medieval old town, with buildings dating from the 13th to 15th centuries. Architecture is characterised by pointed arches, heavy stone construction, narrow facades, and dark narrow streets. Many structures show Catalan Gothic features — wider and lower than northern European Gothic, with emphasis on horizontal lines.",
    'modernista': "Catalan Modernisme (roughly 1888–1930) is Barcelona's distinctive Art Nouveau movement. Buildings feature organic curved forms, floral and natural motifs, ornate wrought iron balconies, ceramic tile decoration, and sculptural facades. Key architects include Antoni Gaudí, Lluís Domènech i Montaner, and Josep Puig i Cadafalch.",
    'olympic_modern': "The Olympic Modern style emerged from Barcelona's 1992 Olympic Games redevelopment, particularly in the Montjuïc and Diagonal Mar areas. Buildings feature contemporary materials — glass, steel, and concrete — with clean geometric lines, large transparent facades, and minimal ornamentation. The style reflects late 20th century international modernism adapted to Barcelona's urban context."
}

def build_model(num_classes: int = 4):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model