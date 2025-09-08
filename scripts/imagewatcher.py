#!/usr/bin/env python
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""\
Monitor a directory for new images.
Not efficient.
"""

import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from datetime import datetime

import pygame

from yapocis.fonts import font as find_font
from yapocis.utils.basic import VIEWING_QUEUE
queue = VIEWING_QUEUE



def get_images():
    images = [os.path.join(queue,name) for name in os.listdir(queue) if name.endswith(".png") or name.endswith(".pgm")]
    images = [(os.stat(path).st_ctime, path) for path in images]
    images.sort()
    return images

most_recent = None
image_index = -1
images = []

def image_path(i=None):
    global image_index
    i = image_index if i is None else i
    images[:] = get_images()
    if len(images) == 0:
        return None, None
    i = i + len(images) if i < 0 else i
    i = i % len(images)
    created, path = images[i]
    return created, path

def fetch_image(i):
    global image_index
    global most_recent
    created, path = image_path(i)
    if created is None or path is None:
        return None,None
    try:
        created = str(datetime.fromtimestamp(created)).split(".")[0]
        image = pygame.image.load(path)
    except pygame.error:
        return None, None
    if max(image.get_size()) > WIDTH:
        image = pygame.transform.smoothscale(image, (WIDTH-50, HEIGHT-50))
    image_name,_ = os.path.splitext(os.path.basename(path))
    image_index = i
    most_recent = path
    return image, f"{i+1}/{len(images)}@{created}  {image_name}"

def next_image():
    return fetch_image(image_index + 1)


def previous_image():
    return fetch_image(image_index - 1)


def last_image():
    return fetch_image(-1)


def first_image():
    return fetch_image(0)


def step_image():
    if image_index < len(images) - 1:
        return next_image()
    if images and last_image and images[-1] != last_image:
        return fetch_image(-1)
    return None, None

def delete_image(i=None):
    global most_recent
    if i is None:
        path = most_recent
    else:
        _, path = image_path(i)
        most_recent = path
    if path and os.path.exists(path):
        os.unlink(path)
    get_images()

WIDTH=1024
HEIGHT=800
def main():
    global WIDTH, HEIGHT
    pygame.init()
    font_path = find_font("default")
    font = pygame.font.Font(font_path, 22)

    background_color = (100, 100, 100)
    display_surface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    pygame.display.set_caption('Image Watcher')
    pygame.key.set_repeat(150)
    image, label_text = next_image()
    while True:
        if image is not None:
            display_surface.fill(background_color)
            w, h = image.get_size()
            display_surface.blit(image, (25 + WIDTH // 2 - w // 2, 25 + HEIGHT // 2 - h // 2))
            if label_text is not None:
                label = font.render(label_text, True, (255, 255, 255), (0, 0, 0))
                display_surface.blit(label, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.VIDEORESIZE:
                # Resize window and image to match
                dim = min(event.w, event.h)
                WIDTH = HEIGHT = dim
                display_surface = pygame.display.set_mode((dim + 50, dim + 50), pygame.RESIZABLE)
                image, label_text = fetch_image(image_index)
            if event.type == pygame.USEREVENT:
                image, label_text = step_image()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                # Stop tailing the directory
                pygame.time.set_timer(pygame.USEREVENT, 0)
                if event.key == pygame.K_a:
                    # Backup
                    image, label_text = previous_image()
                elif event.key == pygame.K_d:
                    # Forward
                    image, label_text = next_image()
                elif event.key == pygame.K_e:
                    # End
                    image, label_text = last_image()
                elif event.key == pygame.K_w:
                    # Beginning
                    image, label_text = first_image()
                elif event.key == pygame.K_s:
                    # Start tailing the directory
                    pygame.time.set_timer(pygame.USEREVENT, 100)
                elif event.key == pygame.K_BACKSPACE:
                    # Delete the current image
                    if event.mod & 1:
                        while True:
                            delete_image()
                            image, label_text = next_image()
                            if image is None:
                                display_surface.fill(background_color)
                                break
                    else:
                        delete_image()
                    image, label_text = next_image()
                    break
        pygame.display.update()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        queue = sys.argv[1]
    main()
