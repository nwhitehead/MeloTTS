import json
import pygame as pg
import sys
import random
import p2v

def main():
    data = json.loads(open('test2.json').read())
    screen = pg.display.set_mode((1024, 768))
    clock = pg.time.Clock()
    talk = pg.mixer.music.load('test2.ogg')
    done = False
    playing = False
    start_time = 0
    state = 'X'
    last_transition_time = 0
    next_state = 'X'
    cue_map = {
        'X': 'lisa-X',
        'A': 'lisa-A',
        'B': 'lisa-B',
        'C': 'lisa-C',
        'D': 'lisa-D',
        'E': 'lisa-E',
        'F': 'lisa-F',
        'G': 'lisa-G',
        'H': 'lisa-H',
    }
    imgs = {
        key: pg.image.load(f'{cue_map[key]}.png')
        for key in cue_map
    }
    delta = 0
    visemes = [ {
        'viseme': cue['value'],
        'time': cue['start'] * 1000 + delta,
    } for cue in data['mouthCues']]
    original_visemes = visemes[:]
    print(visemes)

    while not done:
        screen.blit(imgs[state], (0, 0))
        pg.display.flip()
        if playing:
            time = pg.time.get_ticks() - start_time
            while len(visemes) > 0 and visemes[0]['time'] < time:
                viseme = visemes.pop(0)
                sys.stdout.write(f"{viseme['viseme']} ")
                sys.stdout.flush()
                next_state = viseme['viseme']
            if len(visemes) == 0:
                next_state = 'X'
            if time - last_transition_time > 100:
                state = next_state
                last_transition_time = time

        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            elif event.type == pg.MOUSEBUTTONDOWN:
                start_time = pg.time.get_ticks()
                playing = True
                pg.mixer.music.play()
                visemes = original_visemes[:]
                state = 'X'
                last_transition_time = 0
                next_state = 'X'


if __name__ == '__main__':
    pg.init()
    main()
    pg.quit()
