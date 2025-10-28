def mask_to_yolo(mask_mask):
    res = []
    height, width = mass_mask.shape
    nbr_mass = len(np.unique(mass_mask))-1

    for i in range(nbr_mass):
        mask = mass_mask.copy()
        mask[mass_mask != i+1] = 0
        #her kütlenin konturlarını bulma
        cnts, _= cv2.findContours(mask.astype(np.uint8,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #konturların etrafında bir bbox oluşturma
        x,y,w, h = cv2.boundingRect(cnts[0])
        #yolo formatına dönüştürme
        x = x+w//2 - 1 
        y = y+h//2 - 1
        res.append([x/width,y/height,w/width,h/height, 'mass'])
    return res
#kırpma
def crop(img,mask):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold( blur,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
                   cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w],breast_mask[y:y+h, x:x+w], mask[y:y+h, x:x+w]

    #normalizasyon
    def truncation_normalization(img, mask):
        Pmin = np.percentile(img[mask != 0],5)
        Pmax = np.percentile(img[mask !=0],99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin)/(Pmax-Pmin)
    normalized[mask == 0] = 0
    return normalized