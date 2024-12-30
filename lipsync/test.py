import json
import pygame as pg
import sys
import random
import p2v

def main():
    data = json.loads(open('data.json').read())
    phonemes = []
    screen = pg.display.set_mode((1024, 768))
    clock = pg.time.Clock()
    talk = pg.mixer.music.load('test2.ogg')
    visemes = ['neutral', 'blink', 'ah', 'consonant', 'eee', 'oh', 'w', 'm', 'ch']
    done = False
    playing = False
    start_time = 0
    imgs = [pg.image.load(f'{v}.png') for v in visemes]
    state = 0
    last_transition_time = 0
    next_state = 0

    phoneme_list  = [ph['phoneme'] for ph in data['phonemes']]
    phoneme_times  = [ph['time'] for ph in data['phonemes']]
    phoneme_list = phoneme_list[:-1]
    viseme_list, viseme_times = p2v.to_visemes(phoneme_list, phoneme_times)
    delta = -80
    visemes = [ {
        'viseme': v,
        'time': t + delta,
    } for v, t in zip(viseme_list, viseme_times)]
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
                next_state = -1
                v = viseme['viseme']
                if v in ['sil']:
                    next_state = 0
                if v in ['aa', 'E']:
                    next_state = 2
                if v in ['DD', 'nn', 'kk', 'SS', 'RR']:
                    next_state = 3
                if v in ['I']:
                    next_state = 4
                if v in ['U']:
                    next_state = 6
                if v in ['PP', 'FF']:
                    next_state = 0
                if v in ['CH']:
                    next_state = 8
                if next_state == -1:
                    print(v)
                    next_state = 0
            if len(visemes) == 0:
                next_state = 0
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
                state = 0
                last_transition_time = 0
                next_state = 0


if __name__ == '__main__':
    pg.init()
    main()
    pg.quit()
