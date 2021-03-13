#!/usr/bin/env python
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
"""\
Monitor a directory for new images.
Not efficient.
"""

import os
import pygame
from yapocis.fonts import font as find_font

WIDTH, HEIGHT = 1024, 1024

queue = "/Users/seant/workspace/yapocis/queue"
def get_images():
    images = [os.path.join(queue,name) for name in os.listdir(queue) if name.endswith(".png")]
    images = [(os.stat(path).st_ctime, path) for path in images]
    images.sort()
    return [path for (_, path) in images]

most_recent = None
image_index = -1
images = []

def next_image():
    return fetch_image(image_index+1)

def previous_image():
    return fetch_image(image_index-1)

def last_image():
    return fetch_image(-1)

def first_image():
    return fetch_image(0)

def step_image():
    if image_index < len(images)-1:
        return next_image()
    if images and last_image and images[-1] != last_image:
        return fetch_image(-1)
    return None, None


def fetch_image(i):
    global image_index
    global most_recent
    images[:] = get_images()
    if len(images) == 0:
        return None, None
    i = i + len(images) if i < 0 else i
    i = i % len(images)
    try:
        image = pygame.image.load(images[i])
    except pygame.error:
        return None, None
    if max(image.get_size()) > WIDTH:
        image = pygame.transform.smoothscale(image, (WIDTH, HEIGHT))
    image_name,_ = os.path.splitext(os.path.basename(images[image_index]))
    image_index = i
    most_recent = images[image_index]
    return image, f"{i:3d} {image_name}"

def main():
    pygame.init()
    font_path = find_font("default")
    font = pygame.font.Font(font_path, 32)

    dark = (0, 0, 0)
    display_surface = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption('Image Watcher')

    image, label_text = next_image()
    while True:
        if image is not None:
            display_surface.fill(dark)
            w,h = image.get_size()
            display_surface.blit(image, (WIDTH//2-w//2, HEIGHT//2-h//2))
            if label_text is not None:
                label = font.render(label_text, True, (255, 255, 255))
                display_surface.blit(label, (0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.USEREVENT:
                image, label_text = step_image()
            if event.type == pygame.KEYDOWN:
                pygame.time.set_timer(pygame.USEREVENT, 0)
                if event.key == pygame.K_a:
                    image, label_text = previous_image()
                elif event.key == pygame.K_d:
                    image, label_text = next_image()
                elif event.key == pygame.K_e:
                    image, label_text = last_image()
                elif event.key == pygame.K_w:
                    image, label_text = first_image()
                elif event.key == pygame.K_s:
                    pygame.time.set_timer(pygame.USEREVENT, 100)
        pygame.display.update()


if __name__ == '__main__':
    main()