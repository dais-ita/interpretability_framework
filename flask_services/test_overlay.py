import cv2


img = cv2.imread("test_input.jpg")
input_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img = cv2.imread("test.png",cv2.IMREAD_UNCHANGED)
LRP_image = img#cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

print(input_image.shape)
print(LRP_image.shape)

# explanation_image = input_image[:]
# alpha = 1
# cv2.addWeighted(LRP_image, alpha, explanation_image, 1 - alpha,0, explanation_image)



cv2_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
cv2.imshow("input_image",cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

b,g,r,a=cv2.split(LRP_image)
overlay_color = cv2.merge((b,g,r))
mask=a

cv2.imshow("LRP_image",LRP_image[:,:,:3])
cv2.waitKey(0)
cv2.destroyAllWindows

img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

cv2.imshow("img2_fg",img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows

print("add")
explanation_image = cv2.add(input_image, img2_fg)

cv2_image = cv2.cvtColor(explanation_image, cv2.COLOR_RGB2BGR)
cv2.imshow("explanation_image LRP",cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()