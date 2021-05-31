from model.config import *

transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def image_transform(img_path):
    img = Image.open(img_path)
    img1 = transform1(img)
    return img1


def image_augmentation_transform(img_path):
    img = Image.open(img_path)
    img1 = train_transform(img)
    return img1


def process_image_batch(image_batch, resnet50):
    return resnet50(image_batch)


def show_image(img):
    mode = transforms.ToPILImage()(img)
    plt.imshow(mode)
    plt.show()


